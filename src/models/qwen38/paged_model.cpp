#include "strata/common/serve/request_output.hpp"
#include "strata/base/gpu/managed_memory.hpp"
// src/models/qwen38/paged_model.cpp - strata-q38: Qwen3.8 on the shared paged engine (paged_model.hpp): its start
// and its side of the engine's model interface (strata/common/serve/paged_model.hpp).
#include "strata/models/qwen38/paged_model.hpp"

#include "strata/base/gpu/on_device.hpp"
#include "strata/base/gpu/gfx_arch.hpp"
#include "strata/common/artifact/gguf_reader.hpp"
#include "strata/base/gpu/runtime.hpp"
#include "strata/models/qwen38/prompt_images.hpp"
#include "strata/common/experts/card_rules.hpp"
#include "strata/common/experts/expert_pages.hpp"
#include "strata/common/experts/expert_swaps.hpp"
#include "strata/common/experts/ram_experts.hpp"
#include "strata/common/kernels/cpu/expert_layout.hpp"
#include "strata/common/kernels/mrope.hpp"
#include "strata/common/serve/paged_engine.hpp"
#include "strata/common/serve/paged_model.hpp"
#include "strata/common/serve/penalty_history.hpp"
#include "strata/common/state/checkpoint_retention.hpp"
#include "strata/common/state/state_bytes.hpp"
#include "strata/models/qwen38/rewind_checkpoint.hpp"
#include "strata/common/state/prompt_checkpoint.hpp"
#include "strata/models/qwen38/conversation.hpp"
#include "strata/models/qwen38/conversation_drafter.hpp"
#include "strata/models/qwen38/conversation_pages.hpp"
#include "strata/models/qwen38/conversation_snapshot.hpp"
#include "strata/models/qwen38/kernels/verify_kernels.hpp"
#include "strata/models/qwen38/model.hpp"
#include "strata/models/qwen38/options.hpp"
#include "strata/models/qwen38/session_identity.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using strata::serve::print_response;

namespace strata::qwen38 {
namespace {

using strata::serve::PagedCard;
using strata::serve::PagedCards;
using strata::serve::PagedParts;
using strata::serve::PagedSettings;

/// The program's name, which starts its stderr lines.
const std::string kProgram = "strata-q38";

/// The milliseconds the CPU pool takes for one routed expert of a decode step, from RAM, against which a hop to
/// another card is weighed (--remote-experts auto, card_rules' is_hop_cheaper): the Strix Halo's pool computed 70,349
/// experts in 2,449 ms and 43,808 in 1,550 ms (two stages, 6 requests each, the reads from the SSD not counted).
constexpr double kCpuExpertMs = 0.035;
/// With drafting, the verify windows' graphs kept for the most recent row layouts (the old engine's --batch-mtp).
constexpr size_t kWindowGraphs = 64;
/// The largest chunk a prompt is read in with `--prefill auto` or no --prefill: the old engine's auto ceiling.
constexpr int64_t kPromptCeiling = 8192;

/// Qwen3.8's side of one of the engine's cards (PagedCard of the same index): its prompt path and verify windows,
/// which hand their rows to the next card's.  Card 0 runs on the Model's own cache and session and the Conversation's
/// windows, a later card on one of Model::stages.
struct ModelCard {
    GpuStage* stage = nullptr;                    ///< null for card 0
    /// the hand-offs its verify windows take their rows from (null for card 0) and give them to (null for the last)
    const float* hand_in = nullptr;
    float* hand_out = nullptr;
    strata::core::ExpertCache* cache = nullptr;   ///< the expert cache whose slots are the pool's pages
    strata::core::SessionState* own = nullptr;    ///< the engine's own session on it (the rope table and PLE it lends)
    void* cs = nullptr;                           ///< its stream, as the session functions take it
    strata::core::Verifier* ver = nullptr;
    strata::prefill::Prefill* sp = nullptr;
    int32_t* d_res = nullptr;                     ///< the residency table on its device
    /// computes the experts of other cards' layers its pool holds (PagedCard::lent) that a verify window of their own
    /// card routes to
    strata::core::PeerExperts hop;
};
using ModelCards = std::deque<ModelCard>;   ///< (a deque, as PagedCards)

using Checkpoint = RewindCheckpoint;

/// Holds one card's session and page-pool state; only the lead card owns the conversation's host checkpoints.
struct Part final : strata::serve::PagedPart {
    std::vector<Checkpoint> checks; ///< Only the first card retains the chain shared by all cards.
    bool checkpoint_enabled = true; ///< The current request's ckpt key.
    int64_t pin = 0; ///< The current request's shared prefix, or zero.
    std::vector<int32_t> image_positions; ///< rotary coordinates retained with paused state
    const Model& m;
    const ModelCard& card;              ///< its card's model side, whose stream and verify windows it uses
    int dev = 0;                        ///< its card's device
    strata::core::SessionState ss;
    void* mem = nullptr;                ///< `ss`'s memory (without the state the pages hold)
    ConversationPages pages;            ///< given at the conversation's first request
    /// where its GDN state was when the card's verify windows' graphs last saw it (their graphs hold the address)
    const float* gdn_seen = nullptr;

    bool image_table_ready = false; ///< Whether all cards have this conversation's current coordinates.
    int32_t* image_table = nullptr; ///< Stable coordinates for this card's indexer graph launches.
    Part(const Model& model, const ModelCard& on_card, int device) : m(model), card(on_card), dev(device) {}
    ~Part() override {
        const strata::core::OnDevice on(dev);
        strata::core::gpu_free(image_table);
        strata::core::session_release(ss);
        (void) strata::core::release_device(mem);
    }

    bool is_ready() const override { return pages.is_ready(); }
    bool init(strata::core::PagePool& pool, strata::core::ConversationId id, std::string& err) override {
        if (!pages.init(m.g, ss, pool, id, err)) {
            // Fixed regions may exist before the device tables are ready.
            pool.remove_conversation(id);
            return false;
        }
        return note_gdn_place(err);
    }
    bool reserve(int64_t tokens, std::string& err) override { return pages.reserve(tokens, err); }
    bool can_reserve_without_growth(int64_t tokens) const override {
        return pages.can_reserve_without_growth(tokens);
    }
    void truncate(int64_t tokens) override {
        std::erase_if(checks, [tokens](const Checkpoint& c) { return c.tokens > tokens; });
        pages.truncate(tokens);
    }
    void release() override {
        checks.clear();
        pages.release();
    }
    bool refresh(std::string& err) override { return pages.refresh(ss, err) && note_gdn_place(err); }
    void clear_state() override {
        const strata::core::OnDevice on(dev);
        strata::core::session_zero(ss, m.g, nullptr, card.cs);
        (void) cudaStreamSynchronize((cudaStream_t) card.cs);
    }
    void set_layers(int64_t lb, int64_t le) override {
        strata::core::session_set_layers(ss, m.g, lb, le);
        gdn_seen = nullptr;
    }
    uint64_t bytes() const override { return pages.bytes(); }
    strata::core::ConversationId id() const override { return pages.id(); }
    int64_t get_pool_tokens(int64_t tokens) const override { return pages.get_pool_tokens(tokens); }

    /// Drops the card's verify windows' graphs when its GDN state is elsewhere than when they last saw it.
    bool note_gdn_place(std::string& err) {
        const bool moved = gdn_seen != nullptr && gdn_seen != ss.gdn_state;
        gdn_seen = ss.gdn_state;
        const strata::core::OnDevice on(dev);
        return !moved || card.ver->drop_batch_graphs(err);
    }
};

/// Gets the Part a PagedPart of this model is.
Part& get_part(strata::serve::PagedPart& part) { return static_cast<Part&>(part); }

/// Qwen3.8 for the paged engine: the Model and its Conversation (card 0's prompt path and verify windows), the cards'
/// stages, the layer split's CPU routing and the MTP drafter.
class PagedQwen final : public strata::serve::PagedModel {
public:
    /// `drafter`: null when drafting is off; `sessions`: the conversations the engine holds (its batch slots and
    /// "main").
    PagedQwen(Model& m, Conversation& conv, const PagedCards& cards, ModelCards& model_cards, SplitDrive& split,
              const strata::core::RamExperts& ram, ConversationDrafter* drafter, int sessions, int prompt_rows);

    bool prepare_images(const strata::serve::RequestLine& request, const std::vector<int64_t>& ids,
                        std::vector<strata::core::ConversationImageKey>& images, std::string& err) override;
    void attach_images(const PagedParts& parts) override {
        Part& lead = get_part(*parts.front());
        lead.image_positions = std::move(pending_positions_);
        lead.image_table_ready = false;
    }
    PagedParts make_parts(PagedCard& card, size_t first, size_t count, std::string& err) override;
    bool get_card_layout(const PagedCard& card, int64_t lb, int64_t le, strata::core::StateLayout& layout,
                         std::string& err) const override;
    uint64_t get_gpu_keep(const PagedCard& card) const override;
    std::vector<strata::core::ExpertKey> get_card_ranking(int64_t lb, int64_t le, int64_t& max_experts) const override;
    bool load_expert(const PagedCard& card, strata::core::ExpertKey key, int32_t first, std::string& err) override;
    std::vector<int32_t>& get_residency() override { return m_.host_res; }
    bool publish_residency(const PagedCard& card, std::string& err) override;
    std::vector<float>& get_usage() override { return m_.drive.d.usage; }
    bool set_card_layers(const PagedCard& card, std::string& err) override;
    strata::core::LayerSizes get_layer_sizes(uint64_t page_bytes) const override;
    bool can_run_layers(int64_t lb, int64_t le) const override;
    strata::core::RegionSpan get_gpu_span(const PagedCard& card) const override;
    strata::core::RegionSpan get_ram_span(int64_t lb, int64_t le) const override;
    bool attach_hop(PagedCard& card, std::string& err) override;
    void set_hop_holders(const int32_t* holder) override { split_.holder = holder; }
    void report_hops() const override;
    void start_request(int session, const strata::serve::RequestStart& r) override;
    bool save_state(const PagedParts& parts, std::vector<uint8_t>& state, std::string& err) const override;
    bool restore_state(const std::vector<uint8_t>& state, const PagedParts& parts, const std::vector<int32_t>& live,
                       int64_t kept, std::string& err) const override;
    bool keeps_rewind_branch(const PagedParts& parts, int64_t tokens) const override;
    std::vector<int64_t> get_rewind_points(const PagedParts& parts) const override;
    uint64_t get_host_state_bytes(const PagedParts& parts) const override {
        const auto& checks = get_part(*parts.front()).checks;
        uint64_t bytes = checks.capacity() * sizeof(Checkpoint);
        for (const auto& checkpoint : checks)
            bytes += strata::core::get_owned_vector_bytes(checkpoint.state) +
                     strata::core::get_owned_vector_bytes(checkpoint.recurrent);
        return bytes;
    }
    bool rewind(const PagedParts& parts, int64_t tokens, std::string& err) override;
    bool get_session_identity(strata::core::SessionFileIdentity& id, std::string& err) const override {
        strata::core::SessionIdentityBuilder config(get_session_config_fingerprint(m_));
        // Rewind payloads belong to the header; GPU page sections hold only the active recurrence.
        config.u64("paged state layout", 4);
        config.u64("rewind slots", (uint64_t) m_.o.prompt_cache);
        id.config = config.digest();
        return get_session_model_fingerprint(m_, &m_.src, id.model, err);
    }
    void forget_drafts(int session) override { drafter_->forget(session); }
    int64_t get_prompt_chunk() const override { return conv_.sp.chunk(); }
    int64_t get_prompt_ceiling() const override { return m_.o.prefill_chunk; }
    strata::serve::PromptBuffers get_prompt_buffers(const PagedCard& card, int64_t chunk,
                                                    int64_t tokens) const override;
    bool lay_out_prompt(const PagedCard& card, int64_t chunk, uint8_t* device, uint8_t* host,
                        std::string& err) override;
    bool read_prompt(const PagedParts& parts, const std::vector<int64_t>& ids, int64_t from, int64_t n, bool drafting,
                     const ChunkDone& on_chunk, std::string& err) override;
    void set_step_histories(const std::vector<const std::vector<int32_t>*>& histories) override {
        histories_ = histories;
    }
    bool run_rows(const std::vector<strata::spec::StepRow>& rows, std::vector<int32_t>& picks,
                  std::string& err) override {
        return run_rows(rows, histories_, picks, err);
    }
    int get_pipeline_stages() const override { return (int) cards_.size(); }
    bool prepare_group(int group, const std::vector<strata::spec::StepRow>& rows, std::string& err) override;
    bool launch_group(int stage, int group, std::string& err) override;
    int poll_group(int stage, int group, std::vector<int32_t>& picks, std::string& err) override;
    bool finish_group(int group, std::vector<int32_t>& picks, std::string& err) override;
    bool keep_group(int group, const std::vector<strata::spec::StepRow>& rows, const std::vector<int>& kept,
                    bool drafting, std::string& err) override;
    bool keep_rows(const std::vector<strata::spec::StepRow>& rows, const std::vector<int>& kept, bool drafting,
                   std::string& err) override;
    int propose(strata::serve::PagedPart& last, int session, int32_t, int64_t, int n, int32_t* out,
                std::string& err, strata::spec::DraftContext* draft) override;
    bool switch_drafter(bool on, std::string& err) override;
    bool wait_state(std::string& err) override;
    void print_info(int slots, int spec) const override;
    void print_done(const strata::serve::RequestDone& done) const override;

private:
    /// Copies the small state beside the page-pool regions, independent of the current layer cut.
    bool save_running_state(const PagedParts& parts, std::vector<uint8_t>& state, std::string& err) const;
    /// Restores the small state and the indexer's partially written pooled row.
    bool restore_running_state(const std::vector<uint8_t>& state, const PagedParts& parts,
                               const std::vector<int32_t>& live, std::string& err) const;
    /// Copies whole-model recurrent rows between a RAM checkpoint and the current GPU layer ranges.
    bool copy_recurrent_state(std::vector<uint8_t>& state, const PagedParts& parts, bool to_gpu,
                              std::string& err) const;
    /// Retains a prompt boundary using the old engine's root-pinned checkpoint policy.
    bool save_checkpoint(const PagedParts& parts, const std::vector<int64_t>& ids, int64_t tokens,
                         bool turn, std::string& err);
    /// Reads a segment between checkpoint boundaries through the prompt path or short verify windows.
    bool read_prompt_segment(const PagedParts& parts, const std::vector<int64_t>& ids, int64_t from, int64_t n,
                             bool drafting, const ChunkDone& on_chunk, std::string& err);
    /// Holds the chain layout consumed by both synchronous and pipelined verifier calls.
    struct RowInputs {
        std::array<int, strata::kernels::kVerifyMaxT> sessions{};
        std::array<int32_t, strata::kernels::kVerifyMaxT> tokens{};
        std::array<int64_t, strata::kernels::kVerifyMaxT> positions{};
        int count = 0;
    };
    /// Keeps row arguments, picture coordinates and sampler histories alive through the wave.
    struct Group : RowInputs {
        std::vector<int32_t> picks;
        std::vector<std::vector<int32_t>> histories;
        int32_t* coordinates = nullptr;
        int device = 0;
        ~Group() {
            const strata::core::OnDevice on(device);
            (void) strata::core::release_host(coordinates);
        }
    };
    std::vector<std::unique_ptr<Group>> groups_;
    /// Converts the shared step's chains to the verifier's row arguments.
    static bool get_row_inputs(const std::vector<strata::spec::StepRow>& rows, RowInputs& inputs, std::string& err);
    /// Verifies `rows` in one window. `tails[r]` holds the tokens row r's conversation had read before the row's
    /// window, from which the penalties' recent tokens are taken (get_penalty_histories).
    bool run_rows(const std::vector<strata::spec::StepRow>& rows, const std::vector<const std::vector<int32_t>*>& tails,
                  std::vector<int32_t>& picks, std::string& err);
    /// Notes the LP lines of `picks`, the tokens after `rows`, from `ver`'s head logits (note_logprobs).
    void note_row_logprobs(const strata::core::Verifier& ver, const RowInputs& rows,
                           const std::vector<int32_t>& picks);
    /// Gets this window's rotary coordinates after ensuring its conversations' indexer tables are current.
    bool get_row_coordinates(const std::vector<strata::spec::StepRow>& rows, int32_t* out, std::string& err);
    /// Uploads changed indexer coordinates and, when shared, activates the prompt/drafter table.
    bool activate_images(int session, std::string& err, bool shared = true);
    std::vector<int32_t> pending_positions_;
    std::vector<float> image_rows_;
    std::vector<const float*> image_row_ptrs_;
    std::vector<Part*> leads_;
    std::vector<std::vector<int32_t*>> image_tables_;
    int draft_image_session_ = -1; ///< The conversation whose cell table the drafter currently holds.
    Model& m_;
    Conversation& conv_;   ///< card 0's prompt path and verify windows, shared by every conversation
    const PagedCards& pcards_;
    ModelCards& cards_;
    SplitDrive& split_;
    const strata::core::RamExperts& ram_;
    ConversationDrafter* const drafter_;
    const int sessions_;
    const int prompt_rows_; ///< The legacy verify width used for short prompt reads.
    uint64_t checkpoint_clock_ = 0; ///< Monotonic recency across active and restored chains.
    std::vector<int32_t> picks_;       ///< the last step's token after each row, for the drafter
    std::vector<const std::vector<int32_t>*> histories_; ///< consumed tokens for the current step
    std::vector<int> history_lengths_; ///< each session's penalty window
    std::vector<uint8_t> staging_;     ///< an expert read from the SSD on its way to a GPU (load_expert)
};

std::string get_split_hop_refusal(const Options& o);
std::vector<uint64_t> get_blob_bytes(const Model& m);
PagedCard& add_card(Model& m, Conversation& conv, PagedCards& cards, ModelCards& model_cards, GpuStage* stage);
bool open_pages(Model& m, PagedCard& card, ModelCard& mc, int64_t tokens_min, int64_t gpu_mib, int64_t max_experts,
                std::string& err);
bool open_ram(Model& m, const PagedCards& cards, const ModelCards& model_cards, const PagedSettings& s,
              strata::core::RamExperts& ram, bool& copy_back, std::string& err);
bool init_card_paths(Model& m, Conversation& conv, PagedCards& cards, ModelCards& model_cards, SplitDrive& split,
                     int handoff_rows, std::string& err);
std::string get_unsupported_reason(const Model& m);
/// Selects the single-gfx1151 IQ3_S policy; the verifier applies it only to fully resident batch graphs.
bool should_inline_resident_shared(const Model& m);
void append_size(std::vector<uint8_t>& bytes, uint64_t n);
uint64_t read_size(const std::vector<uint8_t>& bytes, size_t& at);

}  // namespace

int run_paged_engine(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // each protocol line reaches the server as it is printed
    PagedSettings settings;
    if (const int status = strata::serve::take_paged_options(argc, argv, kProgram, settings); status >= 0)
        return status;
    prepare_process(argc, argv);
    Options o;
    bool have_tokens = false;
    if (const int status = parse_options(argc, argv, o, have_tokens); status >= 0) return status;
    const std::string hop_refusal = get_split_hop_refusal(o);
    if (settings.remote_experts == "on" && !hop_refusal.empty()) {
        std::fprintf(stderr, "%s: --remote-experts on: %s; use --remote-experts auto or off\n", kProgram.c_str(),
                     hop_refusal.c_str());
        return 2;
    }
    strata::serve::set_paged_runtime_env(settings, !o.layer_split.empty() && hop_refusal.empty());
    prime_gpu_queues();
    const bool shared_stream_override = std::getenv("STRATA_SH_STREAM") != nullptr;
    if (const int status = apply_device_defaults(o, true); status >= 0) return status;
    const int slots = strata::kernels::kVerifyMaxT;
    if (settings.step.concurrent < 0)
        settings.step.concurrent = std::max(o.batch, 1);
    const auto window_rows = strata::spec::get_window_rows(
        o.spec >= 2 ? o.spec : 4, o.mtp_max_t, (int) settings.suffix_draft, settings.lookup_chain);
    if (settings.step.lookahead < 0)
        settings.step.lookahead = o.mtp.empty() && settings.suffix_draft == 0 && settings.lookup_chain == 0 ?
                                      0 : std::max(window_rows.total - 1, 0);
    if (settings.step.model_lookahead < 0 && window_rows.model > 0)
        settings.step.model_lookahead = window_rows.model - 1;
    if (const std::string why = strata::serve::get_step_settings_problem(settings.step, strata::kernels::kVerifyMaxT);
        !why.empty()) {
        std::fprintf(stderr, "strata-q38: %s\n", why.c_str());
        return 2;
    }
    const bool drafting = settings.step.lookahead > 0 && !o.mtp.empty();
    settings.drafter = settings.drafter && drafting;
    if (settings.step.lookahead > 0 && o.mtp.empty() && settings.suffix_draft == 0 && settings.lookup_chain == 0) {
        std::fprintf(stderr, "strata-q38: --lookahead needs --mtp or --suffix-draft above 0\n");
        return 2;
    }
    // The MTP's weights are loaded only with a lookahead; no K/V streaming or elastic K/V (the pool
    // replaces them); the batch slots are the engine's own conversations, not the old engine's sessions of the whole
    // context; the experts from the model's files, the RAM's pool in place of the RAM copy or the arena
    o.serve = false;
    o.batch = 0;
    if (!drafting)
        o.mtp.clear();
    o.draft.clear();
    o.spec = strata::kernels::kVerifyMaxT;   // the verify windows' room: --rows may change up to it
    o.kv_grow = false;
    o.kv_resident = 0;
    // strata-q38 does not pipeline verify windows across cards; 0 also keeps the drafter on the last card (mtp_first)
    o.pipeline_windows = 0;
    // Paged caches cannot lend the contiguous expert arena that the prompt-buffer planner assumes; the engine lends
    // the prompt paths their buffers from the page pools instead, for each prompt the largest chunk up to --prefill
    // (a ceiling here) that they can lend then
    o.no_prefill_borrow = true;
    if (o.prefill_auto || o.prefill_chunk <= 0)
        o.prefill_chunk = kPromptCeiling;
    o.prefill_auto = false;
    if (o.resident_budget > 0 && settings.ram_mib < 0)
        settings.ram_mib = (int64_t) (o.resident_budget >> 20);
    o.mmap_experts = true;
    o.resident_cpu_experts = o.resident_pin = o.resident_soft = false;
    o.resident_budget = 0;
    if (o.expert_cache_per_layer) {
        std::fprintf(stderr, "strata-q38: --expert-cache-per-layer: its experts share the page pool\n");
        return 2;
    }
    strata::core::session_set_pool_state(true);
    Model m(o);
    m.own_stages = true;
    if (settings.ram_mib >= 0)
        m.ram_pool_bytes = settings.ram_mib << 20;
    m.mtp.set_kv_in_pages(true);
    Conversation conv;
    PagedCards cards;
    ModelCards model_cards;
    const auto get_gpu_mib = [&](size_t card) {
        return settings.gpu_mib.empty() ? -1 : settings.gpu_mib[std::min(card, settings.gpu_mib.size() - 1)];
    };
    m.open_expert_pages = [&](int64_t max_experts, std::string& err) {
        // from here on the experts are read from the files without the file cache
        if (std::string why; !m.src.read_unbuffered(why))
            std::fprintf(stderr, "strata-q38: WARNING: the experts are read through the file cache (%s)\n",
                         why.c_str());
        PagedCard& card = add_card(m, conv, cards, model_cards, nullptr);
        return open_pages(m, card, model_cards[card.index], settings.tokens_min, get_gpu_mib(0), max_experts, err);
    };
    m.open_stage_pages = [&](GpuStage& st, std::string& err) {
        PagedCard& card = add_card(m, conv, cards, model_cards, &st);
        return open_pages(m, card, model_cards[card.index], settings.tokens_min, get_gpu_mib(card.index),
                          o.expert_cache, err);
    };
    if (const int status = m.load(); status != 0) return status;
    if (const int status = m.build_hit_tables(false); status != 0) return status;
    const bool inline_shared = !shared_stream_override && should_inline_resident_shared(m);
    if (inline_shared)
        std::fprintf(stderr, "strata-q38: fully resident IQ3_S windows keep shared projections on the main stream\n");
    for (ModelCard& mc : model_cards) {
        mc.ver->set_resident_shared_inline(inline_shared);
        mc.d_res = mc.stage != nullptr ? mc.stage->d_res : m.d_res;
        mc.ver->set_dynamic_residency(true);
        mc.ver->set_batch_self_commit_default(true);
#if !defined(STRATA_GPU_METAL)
        // Metal's wait_commit does not fence a deferred batch graph.
        mc.ver->set_batch_commit_async(drafting && !m.multi_gpu);
#endif
    }
    if (const std::string why = get_unsupported_reason(m); !why.empty()) {
        std::fprintf(stderr, "strata-q38: %s\n", why.c_str());
        return 2;
    }
    m.finish_expert_tiers();
    if (o.adapt_every > 0 && o.adapt_swaps > 0)   // the routings the engine's adaptive tier counts
        m.drive.d.usage.assign((size_t) (m.g.n_layers * m.g.n_expert), 0.0f);
    std::string err;
    strata::core::RamExperts ram;
    bool copy_back = false;
    if (!open_ram(m, cards, model_cards, settings, ram, copy_back, err)) {
        std::fprintf(stderr, "strata-q38: %s\n", err.c_str());
        return 1;
    }
    m.use_expert_source(ram);
    m.set_prompt_pinned_share();
    m.arm_prompt_cpu_share();
    ConversationDrafter drafter;
    SplitDrive split;
    const int handoff_rows = strata::kernels::kVerifyMaxT *
                             (settings.batch_groups == "1" ? 1 : strata::kernels::kVerifyMaxT);
    if (!init_card_paths(m, conv, cards, model_cards, split, handoff_rows, err) ||
        (drafting && !drafter.init(m, slots, err))) {
        std::fprintf(stderr, "strata-q38: %s\n", err.c_str());
        return 1;
    }
    // Proposals and changing group membership create many layouts; keep only recent captured graphs.
    if (drafting || settings.batch_groups != "1")
        for (const ModelCard& mc : model_cards)
            mc.ver->set_batch_graph_limit(kWindowGraphs);
    if (drafting && !settings.drafter)   // loaded for the pool's layout of the drafter's K/V, switched on later
        m.unload_drafter();
    PagedQwen model(m, conv, cards, model_cards, split, ram, drafting ? &drafter : nullptr, slots,
                    window_rows.total);
    return strata::serve::serve_paged_model(model, cards, ram, copy_back, settings);
}

namespace {

PagedQwen::PagedQwen(Model& m, Conversation& conv, const PagedCards& cards, ModelCards& model_cards,
                     SplitDrive& split, const strata::core::RamExperts& ram, ConversationDrafter* drafter,
                     int sessions, int prompt_rows)
    : m_(m), conv_(conv), pcards_(cards), cards_(model_cards), split_(split), ram_(ram), drafter_(drafter),
      sessions_(sessions), prompt_rows_(prompt_rows) {
    const Options& o = m.o;
    facts.name = kProgram;
    facts.n_layers = m.g.n_layers;
    facts.n_expert = m.g.n_expert;
    facts.context = o.max_context;
    facts.extra_resident_vram = 10ull << 30;
    facts.max_rows = strata::kernels::kVerifyMaxT;
    facts.end_tokens = o.eos_ids;
    facts.has_drafter = drafter != nullptr;
    facts.drafter_label = "MTP";
    facts.kv_quantization = core::qsa_kv_q4() ? "Q4_0" : core::qsa_kv_int8() ? "INT8" : "FP16";
    facts.draft_limits.confidence = (float) o.spec_min_p;
    facts.drafter_start = "--mtp and a --lookahead above 0";
    facts.expert_bytes = get_blob_bytes(m);
    staging_.resize(*std::max_element(facts.expert_bytes.begin(), facts.expert_bytes.end()));
    for (size_t i = 0; i < m.profile_all.size() && (int64_t) i < o.expert_cache; ++i)
        facts.split_ranking.push_back({m.profile_all[i].first, m.profile_all[i].second});
    facts.gpu_experts = o.expert_cache;
    facts.adapt_every = o.adapt_every;
    facts.adapt_swaps = o.adapt_swaps;
    facts.adapt_min_gain = o.adapt_min_gain;
    facts.adapt_decay = o.adapt_decay;
    facts.row_bytes = (uint64_t) strata::kernels::cpu::H * sizeof(float);
    facts.cpu_expert_ms = kCpuExpertMs;
}

PagedParts PagedQwen::make_parts(PagedCard& card, size_t first, size_t count, std::string& err) {
    using namespace strata::core;
    ModelCard& mc = cards_[card.index];
    const OnDevice on(card.dev);
    // every session on the card reads the engine's own session's rope table there
    const QsaState* rope = &mc.own->qsa_states[mc.own->qsa_primary()];
    // (carved for every layer, as the card's own session: a moved cut changes the layers they are for)
    const uint64_t bytes = session_bytes(m_.g, m_.o.max_context, m_.K, 0, -1, false);
    if (first == 0 && !mc.ver->init_slots(std::vector<SessionState*>((size_t) sessions_, mc.own), err))
        return {};
    image_tables_.resize(cards_.size());
    auto& image_tables = image_tables_[card.index];
    PagedParts parts;
    for (size_t i = 0; i < count; ++i) {
        auto part = std::make_unique<Part>(m_, mc, card.dev);
        if (strata::core::allocate_device(&part->mem, bytes,
            {"other", "Device allocations", "Conversation working buffers"}) != cudaSuccess ||
            session_init(m_.g, m_.o.max_context, m_.K, part->mem, part->ss, 0, -1, rope) == 0) {
            char text[160];
            std::snprintf(text, sizeof text, "no room for conversation %zu's session on %s (%.1f MiB)", i,
                          card.id.c_str(), (double) bytes / 1048576.0);
            err = text;
            return {};
        }
        if (m_.o.vision) {
            part->image_table = static_cast<int32_t*>(gpu_alloc_zeroed(m_.mrope_host.size() * sizeof(int32_t)));
            if (!part->image_table) {
                err = "no room for the conversation's image coordinates";
                return {};
            }
            image_tables.push_back(part->image_table);
        }
        session_set_layers(part->ss, m_.g, card.lb, card.le);
        PleRun& ple = part->ss.ple;   // the engine's PLE weights and buffers, with this conversation's history
        ple = mc.own->ple;
        ple.token = &part->ss.ple_token;
        ple.prev = part->ss.ple_prev;
        ple.hist = part->ss.ple_hist;
        session_zero(part->ss, m_.g, nullptr, mc.cs);
        mc.ver->bind_slot((int) (first + i), &part->ss);
        if (card.index == 0)
            leads_.push_back(part.get());
        parts.push_back(std::move(part));
    }
    if (cudaStreamSynchronize((cudaStream_t) mc.cs) != cudaSuccess) {
        err = "the conversations' sessions on " + card.id + ": " + err;
        return {};
    }
    mc.ver->set_slot_image_tables({image_tables.begin(), image_tables.end()});
    return parts;
}

/// Gets the bytes of one expert of each of `m`'s layers.
std::vector<uint64_t> get_blob_bytes(const Model& m) {
    std::vector<uint64_t> blobs;
    for (int64_t l = 0; l < m.g.n_layers; ++l)
        blobs.push_back(strata::kernels::cpu::expert_layout().blob_bytes(l));
    return blobs;
}

/// Gets whether `card` runs `m`'s MTP drafter (--mtp), which Model::load_drafter puts on a layer split's last card.
bool has_drafter(const Model& m, const ModelCard& card) {
    return !m.o.mtp.empty() && card.stage == m.get_drafter_stage();
}

/// Gets the experts of the expert profile of layers [lb, le), hottest first.  With a layer split, `max_experts` (the
/// most the GPUs hold: --expert-cache) becomes how many of them are among the profile's first `max_experts`, so that
/// the cards hold the experts one GPU would; without one, they are CUDA0's profile and `max_experts` stays.
std::vector<strata::core::ExpertKey> get_card_ranking(const Model& m, int64_t lb, int64_t le, int64_t& max_experts) {
    std::vector<strata::core::ExpertKey> ranking;
    if (m.profile_all.empty()) {
        for (const auto& [layer, expert] : m.profile)
            ranking.push_back({layer, expert});
        return ranking;
    }
    int64_t held = 0;
    for (size_t i = 0; i < m.profile_all.size(); ++i)
        if (const auto [layer, expert] = m.profile_all[i]; layer >= lb && layer < le) {
            ranking.push_back({layer, expert});
            held += (int64_t) i < max_experts;
        }
    max_experts = held;
    return ranking;
}

/// Gets the regions a conversation keeps in the pool of `card` (whose pool is open; `mc` its model side) when the card
/// runs layers [lb, le): ConversationPages::get_layout's for its session carved for them, and the drafter's K/V region
/// of its pool when it runs the drafter (has_drafter).
bool get_card_layout(const Model& m, const PagedCard& card, const ModelCard& mc, int64_t lb, int64_t le,
                     strata::core::StateLayout& layout, std::string& err) {
    strata::core::SessionState carve = *mc.own;   // (a copy of the description only: the memory stays the session's)
    strata::core::session_set_layers(carve, m.g, lb, le);
    if (!ConversationPages::get_layout(m.g, carve, nullptr, layout, err))
        return false;
    if (has_drafter(m, mc))
        layout.regions.push_back(card.experts.pool().layout().regions.back());
    return true;
}

/// Gets the GPU memory `card`'s pool leaves free besides what is allocated there: --vram-reserve-mib, and the drafter's
/// head when the card runs it (has_drafter) and has not bound it.  Before the start binds the head (`started` false),
/// also the head, which the GPU's free memory still holds.  (The prompt path's buffers are lent from the pool.)
uint64_t get_gpu_keep(const Model& m, const ModelCard& card, bool started) {
    const uint64_t head = (card.stage != nullptr ? card.stage->head : m.native_head).row_bytes();
    return ((uint64_t) m.o.vram_reserve_mib << 20) +
           (has_drafter(m, card) && (!started || m.mtp.vram_bytes() == 0) ? m.mtp.bind_bytes(head, m.n_vocab) : 0);
}

/// Gets what card `mc`'s prompt path borrows to read in chunks of `chunk` tokens (Prefill::bytes_needed and
/// host_bytes_needed); `hand_off`: the card hands its rows to a next one.
strata::serve::PromptBuffers get_prompt_bytes(const Model& m, const ModelCard& mc, int64_t chunk, bool hand_off) {
    return {strata::prefill::Prefill::bytes_needed(m.g, *mc.own, chunk, m.srcp != nullptr),
            strata::prefill::Prefill::host_bytes_needed(chunk, hand_off)};
}

/// Gives card `k` its layers in its prompt path and verify windows (PagedCard::lb and le), as their stage of the layer
/// split, handing their rows on to the next card's.
void set_stage(const PagedCards& cards, const ModelCards& model_cards, size_t k) {
    const PagedCard& card = cards[k];
    const ModelCard& mc = model_cards[k];
    const bool last = k + 1 == cards.size();
    mc.sp->set_stage(card.lb, card.le, last ? nullptr : model_cards[k + 1].sp);
    mc.ver->set_stage(card.lb, card.le, mc.hand_in, mc.hand_out);
}

bool PagedQwen::get_card_layout(const PagedCard& card, int64_t lb, int64_t le, strata::core::StateLayout& layout,
                                std::string& err) const {
    return qwen38::get_card_layout(m_, card, cards_[card.index], lb, le, layout, err);
}

uint64_t PagedQwen::get_gpu_keep(const PagedCard& card) const {
    return qwen38::get_gpu_keep(m_, cards_[card.index], true);
}

std::vector<strata::core::ExpertKey> PagedQwen::get_card_ranking(int64_t lb, int64_t le, int64_t& max_experts) const {
    return qwen38::get_card_ranking(m_, lb, le, max_experts);
}

bool PagedQwen::load_expert(const PagedCard& card, strata::core::ExpertKey key, int32_t first, std::string& err) {
    const auto bytes = (int64_t) facts.expert_bytes[(size_t) key.layer];
    // an expert the RAM's pool lacks is read past it, so that the pool keeps its pages for the experts the GPUs lack
    const uint8_t* blob = ram_.holds(key.layer, key.expert) ? m_.srcp->blob_stable(key.layer, key.expert)
                          : m_.srcp->copy_blob(key.layer, key.expert, staging_.data()) ? staging_.data()
                                                                                       : nullptr;
    if (blob == nullptr)
        err = "no bytes for layer " + std::to_string(key.layer) + " expert " + std::to_string(key.expert);
    const strata::core::OnDevice on(card.dev);
    return blob != nullptr && cards_[card.index].cache->fill_slot_blocking(first, blob, err, bytes);
}

bool PagedQwen::publish_residency(const PagedCard& card, std::string& err) {
    const strata::core::OnDevice on(card.dev);
    if (const cudaError_t e = m_.put_residency(cards_[card.index].d_res); e != cudaSuccess) {
        err = "the expert residency table of " + card.id + ": " + cudaGetErrorString(e);
        return false;
    }
    // the adaptive tier counts a prompt's routings too, which costs a download of its ids per layer: only while the
    // card lacks some of its layers' experts, since the tier can move none in otherwise
    const int64_t ne = m_.g.n_expert;
    const bool lacking = !m_.drive.d.usage.empty() &&
                         std::any_of(m_.host_res.begin() + card.lb * ne, m_.host_res.begin() + card.le * ne,
                                     [](int32_t slot) { return slot < 0; });
    auto& on_routes = cards_[card.index].sp->on_routes;
    if (!lacking) {
        on_routes = nullptr;
        return true;
    }
    on_routes = [this](int64_t layer, const int32_t* ids, int64_t n) {
        const float per =
            strata::core::get_prompt_routing_weight(strata::core::kPromptRoutings, m_.g.n_expert_used, n);
        for (int64_t i = 0; i < n; ++i)
            if (ids[i] >= 0 && ids[i] < m_.g.n_expert)
                m_.drive.d.usage[(size_t) (layer * m_.g.n_expert + ids[i])] += per;
    };
    return true;
}

bool PagedQwen::set_card_layers(const PagedCard& card, std::string& err) {
    ModelCard& mc = cards_[card.index];
    if (mc.stage != nullptr) {
        mc.stage->lb = card.lb;
        mc.stage->le = card.le;
    }
    strata::core::session_set_layers(*mc.own, m_.g, card.lb, card.le);
    set_stage(pcards_, cards_, card.index);
    split_.end[card.index] = card.le;
    m_.split_at.clear();   // (complete once the last card has its layers)
    for (size_t k = 1; k < pcards_.size(); ++k)
        m_.split_at.push_back(pcards_[k].lb);
    return mc.ver->drop_batch_graphs(err);
}

strata::core::LayerSizes PagedQwen::get_layer_sizes(uint64_t page) const {
    const strata::core::ModelGeometry& g = m_.g;
    const std::vector<strata::core::StateRegion>& whole = ram_.pages().pool().layout().regions;   // (every layer's)
    strata::core::LayerSizes sizes;
    constexpr size_t fixed = 1;
    for (int64_t l = 0, qsa = 0; l < g.n_layers; ++l) {
        const bool is_qsa = is_qsa_layer(g, l);
        sizes.dense.push_back(0);   // (outside the pools)
        sizes.expert.push_back((strata::kernels::cpu::expert_layout().blob_bytes(l) + page - 1) / page * page);
        sizes.fixed_state.push_back(is_qsa ? 0 : whole[0].unit_bytes / (uint64_t) g.n_gdn_layers());
        sizes.token_bytes.push_back(is_qsa
            ? whole[fixed + (size_t) qsa++].unit_bytes / whole[fixed].unit_tokens() : 0);
    }
    return sizes;
}

bool PagedQwen::can_run_layers(int64_t lb, int64_t le) const {
    // a card needs a GDN layer and a QSA layer
    const int64_t interval = std::max<int64_t>(m_.g.qsa_interval, 1), qsa = le / interval - lb / interval;
    return qsa != 0 && qsa != le - lb;
}

strata::core::RegionSpan PagedQwen::get_gpu_span(const PagedCard& card) const {
    strata::core::RegionSpan span;
    span.first = 1;
    span.ranges.push_back({0, 0, card.experts.pool().layout().regions[0].unit_bytes});
    return span;
}

strata::core::RegionSpan PagedQwen::get_ram_span(int64_t lb, int64_t le) const {
    const strata::core::ModelGeometry& g = m_.g;
    const int64_t interval = std::max<int64_t>(g.qsa_interval, 1), q0 = lb / interval, q1 = le / interval;
    const uint64_t layer = ram_.pages().pool().layout().regions[0].unit_bytes / (uint64_t) g.n_gdn_layers();
    constexpr size_t fixed = 1;
    strata::core::RegionSpan span;
    span.first = fixed + (size_t) q0;
    span.count = le == g.n_layers ? SIZE_MAX : (size_t) (q1 - q0);
    span.ranges.push_back({0, (uint64_t) (lb - q0) * layer, (uint64_t) (le - q1 - (lb - q0)) * layer});
    return span;
}

bool PagedQwen::attach_hop(PagedCard& card, std::string& err) {
    ModelCard& mc = cards_[card.index];
    if (!mc.hop.attach(card.dev, *mc.cache, card.lent.data(), m_.g.n_layers, m_.g.n_expert, err))
        return false;
    split_.hop[card.index] = &mc.hop;
    return true;
}

void PagedQwen::report_hops() const {
    if (m_.o.stats) {
        const auto& d = m_.drive.d;
        std::fprintf(stderr, "strata-q38: profile totals: cpu_misses=%lld gpu_hits=%lld cpu_run=%.3f "
                             "plan=%.3f jobs=%.3f ms\n", (long long) d.multi_misses,
                     (long long) d.cache_hits, d.ms_run, d.ms_plan, d.ms_jobs);
        std::fprintf(stderr, "strata-q38: file totals: reads=%lld bytes=%llu time=%.3f ms\n",
                     (long long) m_.src.reads(), (unsigned long long) m_.src.file_read_bytes(), m_.src.file_ms());
        for (const ModelCard& card : cards_) {
            const auto& v = *card.ver;
            std::fprintf(stderr, "strata-q38: verify totals: windows=%lld wait=%.3f pool=%.3f host=%.3f "
                                 "commit=%.3f ms\n", (long long) v.windows, v.ms_wait, v.ms_pool,
                         v.ms_host, v.ms_commit);
            const std::string profile = card.ver->profile_report();
            if (!profile.empty())
                std::fprintf(stderr, "%s\n", profile.c_str());
        }
    }
    for (size_t k = 0; k < cards_.size(); ++k)
        if (const strata::core::PeerExperts& hop = cards_[k].hop; hop.experts() > 0)
            std::fprintf(stderr, "strata-q38: so far %s computed %lld experts of other cards' layers (%lld rows) in "
                                 "one hop each layer, %.1f ms waited for\n",
                         pcards_[k].id.c_str(), (long long) hop.experts(), (long long) hop.entries(), hop.ms_wait);
}

bool PagedQwen::prepare_images(const strata::serve::RequestLine& request, const std::vector<int64_t>& ids,
                                std::vector<strata::core::ConversationImageKey>& images, std::string& err) {
    pending_positions_.clear();
    image_rows_.clear();
    image_row_ptrs_.clear();
    if (!request.pictures)
        return true;
    if (!m_.o.vision) {
        err = "this engine was started without --vision";
        return false;
    }
    pending_positions_.resize(m_.mrope_host.size());
    std::vector<uint64_t> keys;
    for (const auto& [name, value] : request.keys)
        if (name == "img_keys")
            keys = strata::serve::parse_hex_list(value);
    return read_prompt_images(request.file, ids, m_.g.n_embd, keys, pending_positions_, image_rows_, image_row_ptrs_,
                              images, err);
}

bool PagedQwen::activate_images(int session, std::string& err, bool shared) {
    if (!m_.o.vision)
        return true;
    Part& lead = *leads_[(size_t) session];
    if (!shared && lead.image_table_ready)
        return true;
    const auto& positions = lead.image_positions.empty() ? m_.mrope_host : lead.image_positions;
    const size_t bytes = positions.size() * sizeof(int32_t);
    // Captured consumers use nonblocking streams, so the cached table must arrive before they launch.
    for (size_t k = 0; k < cards_.size(); ++k) {
        const strata::core::OnDevice on(pcards_[k].dev);
        int32_t* table = k == 0 ? m_.d_mrope : cards_[k].stage->mrope;
        strata::kernels::mrope_table_set(table);
        int32_t* cached = image_tables_[k][(size_t) session];
        if ((!lead.image_table_ready && !strata::core::gpu_copy(cached, positions.data(), bytes)) ||
            (shared && (!strata::core::gpu_copy_2d_async(table, bytes, cached, bytes, bytes, 1, nullptr) ||
                        !strata::core::gpu_stream_synchronize(nullptr)))) {
            err = "the image position upload failed";
            return false;
        }
    }
    lead.image_table_ready = true;
    if (shared)
        draft_image_session_ = session;
    return true;
}

void PagedQwen::start_request(int session, const strata::serve::RequestStart& r) {
    strata::kernels::SamplerParams sp = get_sampler_params(r.sampling);
    if (history_lengths_.size() < (size_t) sessions_)
        history_lengths_.resize((size_t) sessions_);
    history_lengths_[(size_t) session] = std::min(sp.penalty_last_n, 4096);
    conv_.ver.set_slot_sampling(session, sp);
    Part& lead = *leads_[(size_t) session];
    lead.checkpoint_enabled = true;
    lead.pin = 0;
    for (const auto& [key, value] : r.keys) {
        if (key == "ckpt")
            lead.checkpoint_enabled = std::atoi(value.c_str()) != 0;
        else if (key == "pin" && m_.o.prompt_cache >= 3)
            lead.pin = std::max<int64_t>(0, std::atoll(value.c_str()));
    }
    if (lead.pin > 0)
        for (Checkpoint& c : lead.checks)
            c.pinned = c.tokens == lead.pin;
}

// The state beside the pages is every QSA layer's indexer state (tail, dead, block_pos) in the layers' order, its
// size first, then the PLE history of the card that owns layer 1, its size first.
// Image rotary coordinates, when present, follow the history.
bool PagedQwen::save_running_state(const PagedParts& parts, std::vector<uint8_t>& state, std::string& err) const {
    strata::core::ConversationStateSizes z;
    if (!strata::core::conversation_state_sizes(m_.g, z, err))
        return false;
    std::vector<uint8_t> qsa, ple;
    for (const std::unique_ptr<strata::serve::PagedPart>& part : parts) {
        const Part& p = get_part(*part);
        const strata::core::OnDevice on(p.dev);
        std::vector<uint8_t> one;   // the PLE history, then each of the card's QSA layers' (tail, dead, block_pos)
        if (!strata::core::conversation_unpaged_save(p.ss, m_.g, one, err))
            return false;
        const size_t n = one.size() - (size_t) p.ss.qsa_alloc * (z.tail + z.dead + z.block_pos);
        if (p.ss.layer_lo <= 1 && p.ss.layer_hi > 1) {
            append_size(ple, n);
            ple.insert(ple.end(), one.begin(), one.begin() + (std::ptrdiff_t) n);
        }
        qsa.insert(qsa.end(), one.begin() + (std::ptrdiff_t) n, one.end());
    }
    state.clear();
    append_size(state, qsa.size());
    state.insert(state.end(), qsa.begin(), qsa.end());
    state.insert(state.end(), ple.begin(), ple.end());
    const auto& positions = get_part(*parts.front()).image_positions;
    if (!positions.empty()) {
        append_size(state, positions.size());
        const auto* bytes = reinterpret_cast<const uint8_t*>(positions.data());
        state.insert(state.end(), bytes, bytes + positions.size() * sizeof(int32_t));
    }
    return true;
}

bool PagedQwen::restore_running_state(const std::vector<uint8_t>& state, const PagedParts& parts,
                              const std::vector<int32_t>& live, std::string& err) const {
    strata::core::ConversationStateSizes z;
    if (!strata::core::conversation_state_sizes(m_.g, z, err))
        return false;
    const size_t per_layer = z.tail + z.dead + z.block_pos;
    size_t at = 0;
    const size_t qsa = read_size(state, at);
    if (qsa > state.size() - at || qsa != (size_t) m_.g.n_qsa_layers() * per_layer) {
        err = "the paused conversation's QSA state is damaged";
        return false;
    }
    const auto qsa_begin = state.begin() + (std::ptrdiff_t) at;
    at += qsa;
    const size_t n = read_size(state, at);
    if (n > state.size() - at) {
        err = "the paused conversation's PLE state is damaged";
        return false;
    }
    const size_t ple_at = at;
    at += n;
    for (const std::unique_ptr<strata::serve::PagedPart>& part : parts) {
        Part& p = get_part(*part);
        const strata::core::OnDevice on(p.dev);
        if ((size_t) (p.ss.qsa_ord0 + p.ss.qsa_alloc) * per_layer > qsa) {
            err = "the paused conversation's running state is damaged";
            return false;
        }
        std::vector<uint8_t> one(state.begin() + (std::ptrdiff_t) ple_at,
                                 state.begin() + (std::ptrdiff_t) (ple_at + n));
        const auto first = qsa_begin + (std::ptrdiff_t) ((size_t) p.ss.qsa_ord0 * per_layer);
        one.insert(one.end(), first, first + (std::ptrdiff_t) ((size_t) p.ss.qsa_alloc * per_layer));
        if (!strata::core::conversation_unpaged_restore(one, p.ss, m_.g, live, err))
            return false;
    }
    Part& lead = get_part(*parts.front());
    lead.image_table_ready = false;
    auto& positions = lead.image_positions;
    positions.clear();
    if (at < state.size()) {
        const uint64_t n = read_size(state, at);
        if (!m_.o.vision || n == 0 || n % 3 != 0 || n > (state.size() - at) / sizeof(int32_t) ||
            at + n * sizeof(int32_t) != state.size()) {
            err = "the paused conversation's image positions are invalid";
            return false;
        }
        // The table has a cell per position of the context it was saved with (open_pause_file checked that the
        // tokens fit this one): past the prompt each cell is the previous one plus 1 (read_prompt_images), so a
        // longer context continues that run.
        positions.resize(m_.mrope_host.size());
        const size_t kept = std::min((size_t) n, positions.size());
        std::memcpy(positions.data(), state.data() + at, kept * sizeof(int32_t));
        for (size_t i = kept; i < positions.size(); ++i)
            positions[i] = positions[i - 3] + 1;
    }
    return true;
}

bool PagedQwen::save_state(const PagedParts& parts, std::vector<uint8_t>& state, std::string& err) const {
    std::vector<uint8_t> running;
    if (!save_running_state(parts, running, err))
        return false;
    state.clear();
    append_size(state, running.size());
    state.insert(state.end(), running.begin(), running.end());
    write_rewind_checkpoints(state, get_part(*parts.front()).checks);
    return true;
}

bool PagedQwen::restore_state(const std::vector<uint8_t>& state, const PagedParts& parts,
                              const std::vector<int32_t>& live, int64_t kept, std::string& err) const {
    strata::core::StateReader reader{state};
    std::vector<uint8_t> running;
    std::vector<Checkpoint> checks;
    reader.take_vector(running);
    strata::core::ConversationStateSizes sizes;
    if (!strata::core::conversation_state_sizes(m_.g, sizes, err))
        return false;
    if (!read_rewind_checkpoints(reader, (uint64_t) m_.o.prompt_cache, live.size(), sizes.gdn, checks) ||
        !reader.ok || reader.at != state.size()) {
        err = "the paused conversation's rewind directory is damaged";
        return false;
    }
    // (the running state after all of `live` has dead rows that would be written past the pages of `kept` tokens)
    if ((size_t) kept == live.size() && !restore_running_state(running, parts, live, err))
        return false;
    get_part(*parts.front()).checks = std::move(checks);
    return true;
}

bool PagedQwen::keeps_rewind_branch(const PagedParts& parts, int64_t tokens) const {
    const auto& checks = get_part(*parts.front()).checks;
    return std::none_of(checks.begin(), checks.end(), [tokens](const Checkpoint& c) {
        return c.tokens == tokens && c.turn && !c.pinned;
    });
}

std::vector<int64_t> PagedQwen::get_rewind_points(const PagedParts& parts) const {
    std::vector<int64_t> points;
    for (const Checkpoint& c : get_part(*parts.front()).checks)
        points.push_back(c.tokens);
    return points;
}

bool PagedQwen::rewind(const PagedParts& parts, int64_t tokens, std::string& err) {
    auto& checks = get_part(*parts.front()).checks;
    const auto found = std::find_if(checks.begin(), checks.end(), [tokens](const Checkpoint& c) {
        return c.tokens == tokens;
    });
    if (found == checks.end()) {
        err = "no checkpoint at token " + std::to_string(tokens);
        return false;
    }
    if (!wait_state(err))
        return false;
    if (!copy_recurrent_state(found->recurrent, parts, true, err))
        return false;
    // The restore needs the token count for the indexer and the final two IDs for the PLE history.
    size_t at = 0;
    const int32_t previous = (int32_t) read_size(found->state, at);
    const int32_t last = (int32_t) read_size(found->state, at);
    std::vector<int32_t> ids((size_t) tokens, 0);
    ids.back() = last;
    if (tokens > 1)
        ids[ids.size() - 2] = previous;
    const std::vector<uint8_t> state(found->state.begin() + (std::ptrdiff_t) at, found->state.end());
    if (!restore_running_state(state, parts, ids, err))
        return false;
    for (const Checkpoint& c : checks)
        checkpoint_clock_ = std::max(checkpoint_clock_, c.used);
    found->used = ++checkpoint_clock_;
    return true;
}

bool PagedQwen::copy_recurrent_state(std::vector<uint8_t>& state, const PagedParts& parts, bool to_gpu,
                                     std::string& err) const {
    strata::core::ConversationStateSizes sizes;
    if (!strata::core::conversation_state_sizes(m_.g, sizes, err))
        return false;
    if (!to_gpu)
        state.resize(sizes.gdn);
    if (state.size() != sizes.gdn) {
        err = "the checkpoint's recurrent state has the wrong size";
        return false;
    }
    const size_t layer_bytes = sizes.gdn / (size_t) m_.g.n_gdn_layers();
    for (const auto& part : parts) {
        Part& p = get_part(*part);
        const strata::core::OnDevice on(p.dev);
        void* host = state.data() + (size_t) p.ss.gdn_ord0 * layer_bytes;
        const size_t bytes = (size_t) p.ss.gdn_alloc * layer_bytes;
        if (!strata::core::gpu_copy(to_gpu ? p.ss.gdn_state : host, to_gpu ? host : p.ss.gdn_state, bytes)) {
            err = "the checkpoint's recurrent state could not be copied";
            return false;
        }
    }
    return true;
}

bool PagedQwen::save_checkpoint(const PagedParts& parts, const std::vector<int64_t>& ids, int64_t tokens,
                                 bool turn, std::string& err) {
    auto& checks = get_part(*parts.front()).checks;
    if (std::any_of(checks.begin(), checks.end(), [tokens](const Checkpoint& c) { return c.tokens == tokens; }))
        return true;
    if (!wait_state(err))
        return false;
    Checkpoint saved;
    saved.tokens = tokens;
    saved.turn = turn;
    saved.pinned = tokens == get_part(*parts.front()).pin;
    for (const Checkpoint& c : checks)
        checkpoint_clock_ = std::max(checkpoint_clock_, c.used);
    saved.used = ++checkpoint_clock_;
    append_size(saved.state, tokens > 1 ? (uint64_t) ids[(size_t) tokens - 2] : UINT64_MAX);
    append_size(saved.state, (uint64_t) ids[(size_t) tokens - 1]);
    std::vector<uint8_t> state;
    if (!save_running_state(parts, state, err))
        return false;
    saved.state.insert(saved.state.end(), state.begin(), state.end());
    if (!copy_recurrent_state(saved.recurrent, parts, false, err))
        return false;
    if (checks.size() == (size_t) m_.o.prompt_cache) {
        std::vector<uint64_t> stamps;
        std::vector<int64_t> lengths;
        auto pinned = std::make_unique<bool[]>(checks.size() + 1);
        for (size_t i = 0; i < checks.size(); ++i) {
            stamps.push_back(checks[i].used);
            lengths.push_back(checks[i].tokens);
            pinned[i] = checks[i].pinned;
        }
        pinned[checks.size()] = saved.pinned;
        stamps.push_back(saved.used);
        lengths.push_back(tokens);
        const size_t victim = m_.o.prompt_cache_spread
            ? strata::core::conv_cache::spread_victim(lengths.data(), stamps.data(), stamps.size(),
                                                      m_.o.prompt_cache, nullptr, pinned.get())
            : strata::core::conv_cache::eviction_victim(stamps.data(), stamps.size(), m_.o.prompt_cache,
                                                        nullptr, pinned.get());
        if (victim == checks.size())
            return true;
        checks.erase(checks.begin() + (std::ptrdiff_t) victim);
    }
    checks.push_back(std::move(saved));
    return true;
}

bool PagedQwen::read_prompt(const PagedParts& parts, const std::vector<int64_t>& ids, int64_t from, int64_t n,
                            bool drafting, const ChunkDone& on_chunk, std::string& err) {
    const Part& lead = get_part(*parts.front());
    const bool enabled = m_.o.prompt_cache > 0 && lead.checkpoint_enabled;
    const int64_t end = from + n;
    const int64_t turn = enabled ? strata::core::find_prompt_turn_boundary(
        ids, 0, m_.o.turn_token, m_.o.tail_role_token) : -1;
    const int64_t root = enabled ? strata::core::find_prompt_root_boundary(
        ids, turn, m_.o.turn_token, m_.o.prompt_cache_root) : -1;
    std::vector<int64_t> cuts;
    for (const int64_t point : {root, turn, enabled ? lead.pin : 0})
        if (point > from && point <= end && point + 1 < (int64_t) ids.size())
            cuts.push_back(point);
    if (enabled && m_.o.prompt_cache_every > 0)
        for (int64_t point = (from / m_.o.prompt_cache_every + 1) * m_.o.prompt_cache_every;
             point <= end && (turn < 0 || point < turn); point += m_.o.prompt_cache_every)
            cuts.push_back(point);
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
    int64_t at = from;
    for (int64_t point : cuts) {
        if (!read_prompt_segment(parts, ids, at, point - at, drafting, on_chunk, err) ||
            !save_checkpoint(parts, ids, point, point == turn, err))
            return false;
        at = point;
    }
    return at == end || read_prompt_segment(parts, ids, at, end - at, drafting, on_chunk, err);
}

bool PagedQwen::read_prompt_segment(const PagedParts& parts, const std::vector<int64_t>& ids, int64_t from, int64_t n,
                            bool drafting, const ChunkDone& on_chunk, std::string& err) {
    if (!activate_images((int) get_part(*parts.front()).id() - 1, err))
        return false;
    conv_.sp.embd_rows = image_row_ptrs_.empty() ? nullptr : image_row_ptrs_.data();
    for (size_t k = 0; k < cards_.size(); ++k)
        cards_[k].sp->set_session(get_part(*parts[k]).ss);
    std::vector<int32_t> next;   // the drafter's: the token after each of the chunk's
    strata::prefill::Prefill& last = *cards_.back().sp;   // which has the head
    if (n <= m_.o.short_read && image_row_ptrs_.empty()) {
        if (drafting && !drafter_->preserve_pending_rows(cards_.back().ver->final_R_all(), err))
            return false;
        const int session = (int) get_part(*parts.front()).id() - 1;
        int64_t turn = from + n;
        if (m_.o.prompt_cache > 0 && m_.o.turn_token >= 0)
            for (int64_t i = from + n - 1; i > from; --i)
                if (ids[(size_t) i] == m_.o.turn_token) {
                    turn = i;
                    break;
                }
        for (int64_t pos = from; pos < from + n;) {
            const int64_t end = pos < turn ? turn : from + n;
            const int count = (int) std::min<int64_t>(prompt_rows_, end - pos);
            std::vector<strata::spec::StepRow> rows;
            std::vector<int32_t> picks((size_t) count);
            next.assign(ids.begin() + pos + 1, ids.begin() + pos + count + 1);
            for (int t = 0; t < count; ++t)
                rows.push_back({session, (int32_t) ids[(size_t) (pos + t)], pos + t, t - 1});
            const int64_t recent = history_lengths_[(size_t) session];
            const std::vector<int32_t> tail(ids.begin() + std::max<int64_t>(pos - recent, 0), ids.begin() + pos);
            if (!run_rows(rows, std::vector<const std::vector<int32_t>*>(rows.size(), &tail), picks, err) ||
                !keep_rows(rows, {count}, false, err))
                return false;
            Part& part = get_part(*parts.back());
            if (drafting && !drafter_->read_prompt(nullptr, part.pages, part.ss, cards_.back().ver->final_R_all(),
                    next.data(), count, pos, (int64_t) ids.size(), err))
                return false;
            if (!on_chunk(pos, count, err))
                return false;
            pos += count;
        }
        return true;
    }
    last.on_chunk = [&](const float* R_rows, int64_t T, int64_t pos0, std::string& e) -> bool {
        if (drafting) {
            next.assign(ids.begin() + pos0 + 1, ids.begin() + pos0 + T + 1);
            Part& part = get_part(*parts.back());
            if (!drafter_->read_prompt(&last, part.pages, part.ss, R_rows, next.data(), T, pos0, (int64_t) ids.size(),
                                       e))
                return false;
        }
        return on_chunk(pos0, T, e);
    };
    const bool ok = conv_.read_prompt(ids.data() + from, n, from, err);
    last.on_chunk = nullptr;
    return ok;
}

bool PagedQwen::run_rows(const std::vector<strata::spec::StepRow>& rows,
                         const std::vector<const std::vector<int32_t>*>& tails, std::vector<int32_t>& picks,
                         std::string& err) {
    RowInputs inputs;
    if (!get_row_inputs(rows, inputs, err))
        return false;
    const int S = inputs.count;
    if (m_.o.vision) {
        std::vector<int32_t> coordinates((size_t) S * 3);
        if (!get_row_coordinates(rows, coordinates.data(), err))
            return false;
        for (size_t k = 0; k < cards_.size(); ++k) {
            const strata::core::OnDevice on(pcards_[k].dev);
            int32_t* table = cards_[k].ver->get_batch_image_table();
            if (!strata::core::gpu_copy(table, coordinates.data(), coordinates.size() * sizeof(int32_t))) {
                err = "the step's image position upload failed";
                return false;
            }
        }
    }
    conv_.ver.set_batch_histories(strata::serve::get_penalty_histories(rows, tails, history_lengths_));
    if (!conv_.run_rows(inputs.sessions.data(), S, inputs.tokens.data(), inputs.positions.data(), picks.data(), err))
        return false;
    for (int r = 0; r < S; ++r)
        if (conv_.ver.nonfinite_in(r))
            picks[(size_t) r] = -1;
    picks_ = picks;
    note_row_logprobs(conv_.ver, inputs, picks);
    return true;
}

bool PagedQwen::prepare_group(int group, const std::vector<strata::spec::StepRow>& rows, std::string& err) {
    if (groups_.size() <= (size_t) group)
        groups_.resize((size_t) group + 1);
    if (!groups_[(size_t) group])
        groups_[(size_t) group] = std::make_unique<Group>();
    Group& g = *groups_[(size_t) group];
    if (!get_row_inputs(rows, g, err))
        return false;
    g.histories = strata::serve::get_penalty_histories(rows, histories_, history_lengths_);
    size_t history_tokens = 0;
    for (const auto& history : g.histories)
        history_tokens += history.size();
    if (!cards_.back().ver->reserve_batch_history(history_tokens, err))
        return false;
    g.device = pcards_.back().dev;
    if (drafter_) {
        const strata::core::OnDevice on(g.device);
        if (!drafter_->reserve_snapshots(conv_.ver.final_R_all(), err))
            return false;
    }
    if (m_.o.vision) {
        if (g.coordinates == nullptr && strata::core::allocate_host((void**) &g.coordinates,
            (size_t) strata::kernels::kVerifyMaxT * 3 * sizeof(int32_t), cudaHostAllocPortable, {"other",
                "Device allocations", "Conversation working buffers"}) != cudaSuccess) {
            err = "the group's picture coordinates do not fit";
            return false;
        }
        if (!get_row_coordinates(rows, g.coordinates, err))
            return false;
    }
    for (size_t k = 0; k < cards_.size(); ++k)
        if (!cards_[k].ver->prepare_slot_rows(g.sessions.data(), (int) rows.size(),
            group * strata::kernels::kVerifyMaxT, g.tokens.data(), g.positions.data(), err))
            return false;
    return true;
}

bool PagedQwen::launch_group(int stage, int group, std::string& err) {
    Group& g = *groups_[(size_t) group];
    auto& v = *cards_[(size_t) stage].ver;
    const strata::core::OnDevice on(pcards_[(size_t) stage].dev);
    if (g.coordinates) {
        int32_t* table = v.get_batch_image_table();
        if (cudaMemcpyAsync(table, g.coordinates, (size_t) g.count * 3 * sizeof(int32_t),
                            cudaMemcpyHostToDevice, v.stream()) != cudaSuccess) {
            err = "the pipeline's picture coordinate upload failed";
            return false;
        }
    }
    v.set_batch_histories(g.histories, false);
    return v.launch_slot_rows(g.sessions.data(), g.count, group * strata::kernels::kVerifyMaxT,
                              g.tokens.data(), g.positions.data(), err);
}

int PagedQwen::poll_group(int stage, int, std::vector<int32_t>&, std::string& err) {
    return cards_[(size_t) stage].ver->batch_poll(&drive_pool_split, &split_, err);
}

bool PagedQwen::finish_group(int group, std::vector<int32_t>& picks, std::string& err) {
    Group& g = *groups_[(size_t) group];
    auto& verifier = *cards_.back().ver;
    if (!verifier.restore_slot_commit_inputs(g.sessions.data(), g.count,
            group * strata::kernels::kVerifyMaxT, g.tokens.data(), g.positions.data(), err))
        return false;
    verifier.set_batch_histories(g.histories, false);
    if (!verifier.finish_slot_rows(group * strata::kernels::kVerifyMaxT, picks, err))
        return false;
    g.picks = picks;
    note_row_logprobs(verifier, g, picks);
    return true;
}

bool PagedQwen::keep_group(int group, const std::vector<strata::spec::StepRow>& rows,
                          const std::vector<int>& kept, bool drafting, std::string& err) {
    Group& g = *groups_[(size_t) group];
    std::vector<int> keep((size_t) sessions_, 0);
    for (size_t r = 0, c = 0; r < rows.size(); ++r)
        if (r == 0 || rows[r].session != rows[r - 1].session)
            keep[(size_t) rows[r].session] = kept[c++];
    for (auto& card : cards_) {
        if (!card.ver->restore_slot_commit_inputs(g.sessions.data(), (int) rows.size(),
                group * strata::kernels::kVerifyMaxT, g.tokens.data(), g.positions.data(), err) ||
            !card.ver->commit_slot_prefixes(keep.data(), err, false))
            return false;
    }
    return !drafting || drafter_->note_window(rows, kept, g.picks.data(), err, cards_.back().ver->final_R_all());
}

bool PagedQwen::keep_rows(const std::vector<strata::spec::StepRow>& rows, const std::vector<int>& kept, bool drafting,
                          std::string& err) {
    std::vector<int> keep((size_t) sessions_, 0);
    for (size_t r = 0, c = 0; r < rows.size(); ++r)
        if (r == 0 || rows[r].session != rows[r - 1].session)
            keep[(size_t) rows[r].session] = kept[c++];
    const bool all = rows.size() == kept.size();   // one row each, kept
    // Serial turns must preserve the owned rows of conversations that are waiting for their next turn.
    if (drafting && !drafter_->note_window(rows, kept, picks_.data(), err,
            drafter_->get_snapshot_bytes() ? conv_.ver.final_R_all() : nullptr))
        return false;
    return all ? conv_.ver.commit_slots(err) : conv_.ver.commit_slot_prefixes(keep.data(), err);
}

int PagedQwen::propose(strata::serve::PagedPart& last, int session, int32_t, int64_t, int n, int32_t* out,
                       std::string& err, strata::spec::DraftContext* draft) {
    if (!drafter_->has_pending_rows(session))
        return 0;
    const bool upload = draft_image_session_ != session || !leads_[(size_t) session]->image_table_ready;
    if (!activate_images(session, err, false))
        return -1;
    if (m_.o.vision && upload) {
        const strata::core::OnDevice on(m_.mtp.device());
        int32_t* table = cards_.back().stage ? cards_.back().stage->mrope : m_.d_mrope;
        strata::kernels::mrope_table_set(table);
        const size_t bytes = m_.mrope_host.size() * sizeof(int32_t);
        // Commits read per-conversation tables, so the drafter's shared table can travel on its own stream.
        if (!strata::core::gpu_copy_2d_async(table, bytes, image_tables_.back()[(size_t) session], bytes,
                                            bytes, 1, m_.mtp.stream())) {
            err = "the drafter's image position copy failed";
            return -1;
        }
        draft_image_session_ = session;
    }
    Part& part = get_part(last);
    const int got = drafter_->propose(session, part.pages, part.ss, conv_.ver.final_R_all(), n, out, err, draft);
    if (got < 0)
        (void) wait_state(err);
    return got;
}

bool PagedQwen::get_row_inputs(const std::vector<strata::spec::StepRow>& rows, RowInputs& inputs,
                               std::string& err) {
    if (rows.size() > strata::kernels::kVerifyMaxT) {
        err = "a step of " + std::to_string(rows.size()) + " rows exceeds the verify window";
        return false;
    }
    inputs.count = (int) rows.size();
    for (size_t r = 0; r < rows.size(); ++r) {
        const auto& row = rows[r];
        const bool follows = r > 0 && rows[r - 1].session == row.session;
        if (row.parent != (follows ? (int) r - 1 : -1)) {
            err = "strata-q38's verify windows take each conversation's rows as a chain";
            return false;
        }
        inputs.sessions[r] = row.session;
        inputs.tokens[r] = row.token;
        inputs.positions[r] = row.pos;
    }
    return true;
}

void PagedQwen::note_row_logprobs(const strata::core::Verifier& ver, const RowInputs& rows,
                                  const std::vector<int32_t>& picks) {
    for (int r = 0; r < rows.count; ++r)
        note_logprobs(rows.sessions[(size_t) r], rows.positions[(size_t) r], picks[(size_t) r], ver.vocab(),
                      [&](float* out) { return ver.copy_logits(r, out); });
}

bool PagedQwen::get_row_coordinates(const std::vector<strata::spec::StepRow>& rows, int32_t* out, std::string& err) {
    for (size_t r = 0; r < rows.size(); ++r) {
        if ((r == 0 || rows[r - 1].session != rows[r].session) && !activate_images(rows[r].session, err, false))
            return false;
        const auto& table = leads_[(size_t) rows[r].session]->image_positions;
        for (int axis = 0; axis < 3; ++axis)
            out[r * 3 + axis] = table.empty() ? (int32_t) rows[r].pos : table[(size_t) rows[r].pos * 3 + axis];
    }
    return true;
}

bool PagedQwen::switch_drafter(bool on, std::string& err) {
    if (!on) {
        m_.unload_drafter();
        return true;
    }
    m_.mtp.set_kv_in_pages(true);
    if (!m_.load_drafter(err) || !drafter_->init(m_, sessions_, err)) {
        m_.unload_drafter();
        return false;
    }
    return true;
}

void PagedQwen::print_info(int slots, int spec) const {
    const Options& o = m_.o;
    int64_t resident = 0;
    uint64_t expert_bytes = 0;
    for (size_t k = 0; k < cards_.size(); ++k) {
        resident += cards_[k].cache->resident();
        expert_bytes += pcards_[k].experts.pool().report().experts.bytes;
    }
    std::printf("INFO context=%lld kv=%s expert_slots=%lld expert_cache_mib=%lld spec=%d prefill_cap=%lld%s engine="
                STRATA_VERSION "\n",
                (long long) o.max_context, o.kv.c_str(), (long long) resident, (long long) (expert_bytes >> 20), spec,
                (long long) o.prefill_chunk,
                slots > 0 ? (" batch_slots=" + std::to_string(slots) + " slot_cache=0").c_str() : "");
}

strata::serve::PromptBuffers PagedQwen::get_prompt_buffers(const PagedCard& card, int64_t chunk,
                                                          int64_t tokens) const {
    if (tokens <= m_.o.short_read && image_row_ptrs_.empty())   // (read through the verify windows)
        return {};
    return get_prompt_bytes(m_, cards_[card.index], chunk, card.index + 1 < cards_.size());
}

bool PagedQwen::lay_out_prompt(const PagedCard& card, int64_t chunk, uint8_t* device, uint8_t* host,
                               std::string& err) {
    strata::prefill::Prefill& sp = *cards_[card.index].sp;
    if (chunk == 0)
        return sp.idle(err);
    const strata::serve::PromptBuffers b = get_prompt_bytes(m_, cards_[card.index], chunk,
                                                             card.index + 1 < cards_.size());
    return sp.relayout(chunk, device, b.device, err, host, b.host);
}

void PagedQwen::print_done(const strata::serve::RequestDone& d) const {
    // the fields of the old engine's DONE that this one has (no expert counts)
    print_response("DONE %lld %lld %.1f %.1f %s %lld %lld %lld 0 0 0 0 0.0 %lld 0 chunk=%lld experts_streamed=0\n",
                (long long) d.produced, (long long) d.prompt, d.prompt_ms, d.decode_ms, d.finish,
                (long long) d.guessed_right, (long long) d.guessed, (long long) d.resumed, (long long) d.read,
                (long long) conv_.sp.chunk());
}

/// Gets why the cards of `o`'s layer split may not compute experts in hops, or an empty
/// string: card 0 runs on GPU 0 (a layer split takes no other primary device) and the later cards on --split-device's
/// GPUs, which without the option are GPUs of their own.
std::string get_split_hop_refusal(const Options& o) {
    std::vector<int64_t> stages;
    std::string err;
    if (o.layer_split.empty() || o.split_device.empty() ||
        !strata::serve::parse_i64_list(o.split_device.c_str(), stages, err))
        return {};
    std::vector<int> devices = {0};
    devices.insert(devices.end(), stages.begin(), stages.end());
    return strata::core::get_hop_refusal(devices);
}

/// Adds the card that runs layers [lb, le) of `m` to `cards` and its model side to `model_cards`: CUDA0's (card 0,
/// over the Model's own cache and session and `conv`'s windows) or the layer split's stage `stage`.  Its link to RAM
/// is measured before its pool takes the memory the measuring copies need, and its speed is predicted from its
/// multiprocessors and clock.
PagedCard& add_card(Model& m, Conversation& conv, PagedCards& cards, ModelCards& model_cards, GpuStage* stage) {
    PagedCard& card = cards.emplace_back();
    ModelCard& mc = model_cards.emplace_back();
    card.index = cards.size() - 1;
    card.dev = stage != nullptr ? stage->dev : m.primary_device;
    card.lb = stage != nullptr ? stage->lb : 0;
    card.le = stage != nullptr ? stage->le : m.split_at.empty() ? m.g.n_layers : m.split_at[0];
    const bool shared =
        std::any_of(cards.begin(), cards.end() - 1, [&](const PagedCard& c) { return c.dev == card.dev; });
    card.id = "gpu" + std::to_string(card.dev) + (shared ? "." + std::to_string(card.index) : "");
    mc.stage = stage;
    mc.cache = stage != nullptr ? &stage->cache : &m.xcache;
    mc.own = stage != nullptr ? &stage->ss : &m.ss;
    mc.cs = stage != nullptr ? (void*) stage->stream : m.main_cs;
    mc.ver = stage != nullptr ? &stage->ver : &conv.ver;
    mc.sp = stage != nullptr ? &stage->sp : &conv.sp;
    card.weights_lb = m.stage_weights_trimmed ? card.lb : 0;
    card.weights_le = m.stage_weights_trimmed ? card.le : m.g.n_layers;
    strata::serve::measure_card(card, kProgram, strata::kernels::cpu::expert_layout().max_blob);
    return card;
}

/// Opens `card`'s expert cache (in `mc`) in the pages of its page pool, for at most `max_experts` experts
/// (Model::open_expert_pages, open_stage_pages); with a layer split, for those of the expert profile's first
/// `max_experts` that are of the card's layers, so that the cards hold the experts one GPU would.  The pool takes
/// `gpu_mib` (-1: what get_gpu_room gives, but no more than those experts (the profile's first) and a conversation at
/// --max-context take), in chunks of kChunkBytes; experts leave the pages of `tokens_min` tokens free.  With the
/// drafter (--mtp), the pool of the card that runs it (has_drafter) also keeps each conversation's drafter K/V, and
/// the drafter's head stays out of it.
bool open_pages(Model& m, PagedCard& card, ModelCard& mc, int64_t tokens_min, int64_t gpu_mib, int64_t max_experts,
                std::string& err) {
    const strata::core::OnDevice on(card.dev);
    const std::vector<strata::core::ExpertKey> ranking = get_card_ranking(m, card.lb, card.le, max_experts);
    const bool drafting = has_drafter(m, mc);
    const std::vector<uint64_t> blobs = get_blob_bytes(m);
    const uint64_t page = strata::core::choose_expert_page_bytes(blobs);
    strata::core::StateLayout layout;
    if (!ConversationPages::get_layout(m.g, *mc.own, drafting ? &m.mtp.kv_state() : nullptr, layout, err))
        return false;
    uint64_t most = strata::serve::get_conversation_bytes(layout, m.o.max_context, page);
    for (size_t i = 0; i < ranking.size() && (int64_t) i < max_experts; ++i)
        most += (blobs[(size_t) ranking[i].layer] + page - 1) / page * page;
    uint64_t room = 0;
    const uint64_t keep = get_gpu_keep(m, mc, false);
    const uint64_t pool_b = strata::serve::get_gpu_pool_bytes(card.dev, keep, gpu_mib, most, room);
    const uint64_t chunk = strata::serve::get_chunk_bytes(pool_b, page);
    mc.cache->set_expert_bytes((int64_t) strata::kernels::cpu::expert_layout().max_blob);
    strata::core::PagePoolConfig config;
    config.tokens_min = std::clamp<int64_t>(tokens_min, 0, m.o.max_context);
    strata::core::ChunkMemory memory = strata::serve::get_gpu_chunk_memory(card.dev, chunk, false);
    // with virtual memory the cache's arena is the range's first chunk, so that the pool's other chunks follow it
    uint8_t* arena = memory.contiguous ? memory.alloc(chunk) : nullptr;
    if (memory.contiguous && arena == nullptr) {
        err = "no room for the page pool's first chunk";
        return false;
    }
    if (!mc.cache->open((int64_t) (chunk / page), m.g.n_layers, m.g.n_expert, (int64_t) page, err, arena) ||
        !card.experts.open(*mc.cache, std::move(memory), pool_b, layout, config, blobs, m.g.n_expert, max_experts,
                           ranking, err))
        return false;
    strata::serve::say_gpu_pool(kProgram, card, room, chunk);
    return true;
}

/// Opens the RAM's page pool `ram` beside the cards' (open_ram_pool) and gets whether an expert a GPU's pool evicts is
/// copied back to it (`copy_back`); it takes the profile's experts once the engine serves.  Its conversation layout
/// has the whole model's regions, so that a paused conversation is one conversation in it wherever the layer cut is
/// (PagedQwen::get_ram_span).
bool open_ram(Model& m, const PagedCards& cards, const ModelCards& model_cards, const PagedSettings& s,
              strata::core::RamExperts& ram, bool& copy_back, std::string& err) {
    std::vector<strata::core::ExpertKey> profile;
    for (const auto& [l, e] : m.profile_all.empty() ? m.profile : m.profile_all)
        profile.push_back({l, e});
    strata::core::StateLayout layout;
    const int ring = strata::kernels::kVerifyMaxT * (int) m.K;   // the experts of one layer of a step
    uint64_t prompt = 0;   // every card's prompt path's host buffers
    for (size_t k = 0; k < model_cards.size(); ++k)
        prompt += get_prompt_bytes(m, model_cards[k], m.o.prefill_chunk, k + 1 < model_cards.size()).host;
    return get_card_layout(m, cards.back(), model_cards.back(), 0, m.g.n_layers, layout, err) &&
           strata::serve::open_ram_pool(kProgram, s, m.src, m.o.resident_headroom, ring, layout, cards[0],
                                        get_blob_bytes(m), m.g.n_expert, profile, m.xcache.device_slot(0), prompt,
                                        ram, copy_back, err);
}

/// Gets why strata-q38 cannot run the loaded model `m` (empty: it can).
std::string get_unsupported_reason(const Model& m) {
    if (m.o.native_preset.empty()) return "it needs --native SHARD1 for the dense projections and output head";
    if (m.thits.d_res == nullptr || m.o.expert_cache <= 0)
        return "it needs the device residency table (--expert-profile, --expert-cache and the token graph)";
    return {};
}

/// Sets up each card's prompt path and verify windows over its own session, each handing its rows to the next card's
/// (through hand-offs in page-locked RAM, so that cards without peer access work), and the CPU expert pool's work in
/// the windows going to the card that runs each layer (`split`, which must outlive them).  The Model's stages run in
/// the order of their layers, as the old engine's layer split runs them.
bool init_card_paths(Model& m, Conversation& conv, PagedCards& cards, ModelCards& model_cards, SplitDrive& split,
                     int handoff_rows, std::string& err) {
    const Options& o = m.o;
    const size_t n = cards.size();
    std::vector<float*> hand(n - 1, nullptr);
    if (!alloc_split_handoffs(m.g, handoff_rows, hand)) {
        err = "the layer split's hand-offs do not fit";
        return false;
    }
    for (size_t k = n; k-- > 0;) {   // a card's prompt path is set up after the next card's, which it hands to
        const PagedCard& card = cards[k];
        ModelCard& mc = model_cards[k];
        mc.hand_in = k == 0 ? nullptr : hand[k - 1];
        mc.hand_out = k + 1 == n ? nullptr : hand[k];
        set_stage(cards, model_cards, k);
        // its buffers are lent for each prompt and laid out then (lay_out_prompt); `init` writes none of them, so
        // the pool's first page stands in for them until the first prompt
        mc.sp->set_host_borrowed(true);
        const strata::core::OnDevice on(card.dev);
        uint8_t* stand_in = card.experts.pool().page_address(0);
        const uint64_t bytes = get_prompt_bytes(m, mc, o.prefill_chunk, k + 1 < n).device;
        const bool ok = k == 0 ? conv.init_prompt(m, m.ss, stand_in, bytes, err)
                               : mc.sp->init(mc.stage->wt, m.g, *mc.own, m.srcp, mc.cache, m.host_res.data(),
                                             o.prefill_chunk, mc.cs, err, stand_in, bytes);
        if (!ok) {
            err = card.id + "'s prompt path: " + err;
            return false;
        }
        if (k == 0)
            break;
        const GpuStage& st = *mc.stage;
        if (!mc.ver->init(st.wt, m.g, *mc.own, m.get_verify_hits(st), st.head.loaded() ? &st.head : nullptr, o.spec,
                          err)) {
            err = card.id + "'s windows: " + err;
            return false;
        }
        mc.ver->set_split(o.spec_split);
        mc.ver->set_pcie_mode(get_pcie_mode_code(o));
    }
    if (!conv.init_decode(m, m.ss, o.spec, err))
        return false;
    if (n == 1)
        return true;
    split.base = &m.drive;
    split.n = (int) n;
    for (size_t k = 0; k < n; ++k) {
        const ModelCard& mc = model_cards[k];
        mc.ver->set_handoff_rows(handoff_rows);
        split.end[k] = cards[k].le;
        split.plan[k] = mc.ver->plan_sink();
        split.cache_base[k] = mc.stage != nullptr ? mc.cache->device_slot(0) : m.drive.d.cache_base;
        split.cache_slot_off[k] = mc.stage != nullptr ? mc.cache->slot_offsets() : m.drive.d.cache_slot_off;
        split.pcie_num[k] = get_pcie_num(mc.stage != nullptr ? mc.stage->pcie_frac : o.pcie_frac);
        if (k + 1 < n)
            mc.ver->set_next(model_cards[k + 1].ver, &split);
    }
    conv.set_split_drive(&split);
    return true;
}

/// Appends `n` to `bytes` (save_state's sizes).
void append_size(std::vector<uint8_t>& bytes, uint64_t n) {
    strata::core::append_state_u64(bytes, n);
}

/// Reads a size append_size wrote at `at` in `bytes` and moves `at` past it.
uint64_t read_size(const std::vector<uint8_t>& bytes, size_t& at) {
    uint64_t n = 0;
    if (at > bytes.size() || bytes.size() - at < sizeof n)
        return UINT64_MAX;
    std::memcpy(&n, bytes.data() + at, sizeof n);
    at += sizeof n;
    return n;
}

bool PagedQwen::wait_state(std::string& err) {
    for (const ModelCard& card : cards_)
        if (!card.ver->wait_commit(err))
            return false;
    if (m_.mtp.stream()) {
        const strata::core::OnDevice on(m_.mtp.device());
        if (!strata::core::gpu_stream_synchronize(m_.mtp.stream())) {
            err = "the drafter's state writes did not complete";
            return false;
        }
    }
    return true;
}


bool should_inline_resident_shared(const Model& m) {
#if defined(STRATA_USE_HIP)
    if (m.multi_gpu || !m.native_pack || m.o.native_shards.empty())
        return false;
    cudaDeviceProp device{};
    if (cudaGetDeviceProperties(&device, 0) != cudaSuccess ||
        !strata::kernels::gfx_arch_is_gfx1151(device.gcnArchName))
        return false;
    try {
        const strata::GgufFile file(m.o.native_shards.front());
        const strata::MetaValue* type = file.get("general.file_type");
        return type && type->is_num() && type->u == 26;   // GGUF's IQ3_S tier includes mixed expert formats.
    } catch (const std::exception&) {
        return false;
    }
#else
    (void) m;
    return false;
#endif
}

}  // namespace
}  // namespace strata::qwen38
