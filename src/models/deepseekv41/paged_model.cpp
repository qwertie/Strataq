#include "strata/common/serve/request_output.hpp"
// src/models/deepseekv41/paged_model.cpp - strata-ds41: DeepSeek V4.1 on the shared paged engine (paged_model.hpp):
// its start and its side of the engine's model interface (strata/common/serve/paged_model.hpp).
#include "strata/models/deepseekv41/paged_model.hpp"

#include "ds4_model.hpp"
#include "ds4_images.hpp"
#include "expert_files.hpp"
#include "strata/base/gpu/on_device.hpp"
#include "strata/base/gpu/runtime.hpp"
#include "strata/base/platform/os.hpp"
#include "strata/common/experts/ram_experts.hpp"
#include "strata/common/experts/card_rules.hpp"
#include "strata/common/experts/split_rules.hpp"
#include "strata/common/serve/paged_engine.hpp"
#include "strata/common/serve/paged_model.hpp"
#include "strata/common/state/state_bytes.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using strata::serve::print_response;

namespace strata::ds4 {
namespace {

using strata::serve::PagedCard;
using strata::serve::PagedCards;
using strata::serve::PagedParts;
using strata::serve::PagedSettings;

/// The program's name, which starts its stderr lines.
const std::string kProgram = "strata-ds41";
/// The RAM the RAM pool leaves free at its largest (the old engine's kRamMargin).
constexpr uint64_t kRamHeadroom = 8ull << 30;
/// The top_k of a request that names none (the old engine's).
constexpr int kDefaultTopK = 64;
/// The checkpoints (Model::checkpoint's slots) a prompt's read keeps, as the old engine's does: the state at its end,
/// and before its last token (Model::feed_mark), which Model::pin makes the anchor.
constexpr int kPromptEndCheck = 0;
constexpr int kBeforeLastCheck = 1;
/// The checkpoint of a prompt's start that other requests share (a GEN's pin=N), as the old engine keeps it.
constexpr int kSharedStartCheck = 3;
/// The shortest shared start pin=N keeps a checkpoint of (the old engine's).
constexpr int64_t kShortestPin = 256;
/// The shared step supports at most eight rows, including guesses.
constexpr int kWindowRows = 8;

/// strata-ds41's options beside the paged engine's and the model's: the batch slots, the GPU pool's experts and the
/// adaptive tier (as strata-q38's options of the same names).
struct ProgramOptions {
    int batch = 0;              ///< --batch: the initial number of conversations scheduled together
    int64_t gpu_experts = -1;   ///< --expert-cache: the most experts the GPU's pool holds; -1 (auto): as many as fit
    int adapt_every = 4;        ///< --adapt-every: decode steps between two rounds of the tier
    /// --adapt-swaps: the most experts a round moves into the GPU's pool (0: the tier is off).  The old engine's expert
    /// cache has at most 8 moves under way (kMaxSwaps); a DeepSeek expert is 4 to 5 times a Qwen3.8 one, whose 96 a
    /// round M5b measured
    int adapt_swaps = 8;
    float adapt_min_gain = 1.5f;   ///< --adapt-min-gain: the routings a move must gain (choose_expert_swaps)
    float adapt_decay = 0.7f;      ///< --adapt-decay: the routing counts' factor per 4 decode steps
    std::string layer_split, split_device;  ///< Existing stage bounds and physical devices.
    bool trim_stage_weights = false;       ///< Keeps the cut within the initial dense-weight ranges.
    bool drafting = true;          ///< --dspark is not off: the drafter starts switched on (--drafter)
};

/// A conversation's state on one stage: its regions in the card's pool (describe_conversation's) and the tables of
/// its growing regions' blocks that the kernels read; its host side (tokens, checkpoints), which is in the model while
/// the model computes this conversation (it is seated); and what its request needs of the model.
struct Part final : strata::serve::PagedPart {
    Model& m;
    ConversationRegions at;
    const int dev;                             ///< The physical GPU.
    const int stage;                           ///< The Model's device index, distinct for same-GPU stages too.
    Part* lead = this;                         ///< Owns the conversation host state shared by all its stages.
    std::vector<Part*> peers;                  ///< The lead reconnects every stage when it takes the seat.
    const bool has_drafter;                    ///< Only the last stage keeps DSpark's rings.
    Part*& seated;                             ///< Owns the host side currently in Model, or null.
    strata::core::PagePool* pool = nullptr;     ///< null before init
    strata::core::ConversationId cid = 0;
    /// per region of the layout (empty for a fixed one): the first address of each of its blocks, on the host, and
    /// the device copy the kernels read; sized once for the whole context, so that KvRows may point at them
    std::vector<std::vector<uint8_t*>> blocks;
    std::vector<uint8_t**> tables;
    std::vector<size_t> zeroed;                 ///< per region: its blocks that are zeroed and in the table
    ConversationState state;                    ///< On the lead: its host side while not seated (take_seat).
    strata::serve::Sampling smp;                ///< its request's sampling
    /// the prompt's last token when the prompt's read took it along (feed_mark): its position (-1: none), the token
    /// and the token sampled after it, which the step that would feed it gives instead (another conversation's steps
    /// may have taken the logits since)
    int64_t primed_pos = -1;
    int32_t primed_token = 0, primed_pick = 0;
    uint64_t drafter_epoch = 0; ///< Last dense-weight generation whose rings this conversation rebuilt.
    bool prompt_open = false;   ///< its request's prompt is not read to its end, where it keeps two checkpoints
    /// the end of its next prompt becomes its anchor (Model::pin): its conversation started over or went back
    bool pin_next = false;
    int64_t pin = 0; ///< The prompt prefix kept as a shared checkpoint.

    Part(Model& model, const ConversationRegions& regions, int device, int index, bool drafter, Part*& seated_part)
        : m(model), at(regions), dev(device), stage(index), has_drafter(drafter), seated(seated_part) {}
    ~Part() override {
        if (seated == this)
            seated = nullptr;
        const strata::core::OnDevice on(dev);
        for (uint8_t** t : tables)
            strata::core::gpu_free(t);
    }

    bool is_ready() const override { return pool != nullptr; }
    bool init(strata::core::PagePool& p, strata::core::ConversationId id, std::string& err) override {
        if (const std::string why = p.grow_conversation(id, 0); !why.empty()) {
            err = "the page pool: " + why;
            return false;
        }
        p.set_conversation_label(id, "Conversation " + std::to_string(id));
        pool = &p;
        cid = id;
        return alloc_tables(err) && write_tables(err);
    }
    bool reserve(int64_t tokens, std::string& err) override {
        if (const std::string why = pool->grow_conversation(cid, tokens); !why.empty()) {
            err = "the conversation's memory is full: " + why;
            return false;
        }
        return write_tables(err);
    }
    void truncate(int64_t tokens) override {
        pool->truncate_conversation(cid, tokens);
        for (size_t r = 0; r < blocks.size(); ++r)
            zeroed[r] = std::min(zeroed[r], pool->conversation_blocks(cid, r).size());
    }
    void release() override {
        leave_seat();
        state = {};
        primed_pos = -1;
        pool->remove_conversation(cid);
        pool = nullptr;
        std::fill(zeroed.begin(), zeroed.end(), 0);
    }
    bool refresh(std::string& err) override { return write_tables(err, false); }
    void clear_state() override {
        if (lead != this)
            return;
        take_seat();
        m.reset();
        primed_pos = -1;
        pin_next = true;
    }
    void set_layers(int64_t lb, int64_t le) override {
        const strata::core::OnDevice on(dev);
        for (uint8_t** table : tables)
            strata::core::gpu_free(table);
        tables.clear();
        blocks.clear();
        zeroed.clear();
        strata::core::StateLayout layout;
        std::string err;
        describe_conversation(m.geometry(), (int) lb, (int) le, has_drafter, layout, at, err);
    }
    uint64_t bytes() const override {
        uint64_t bytes = 0;
        for (size_t r = 0; is_ready() && r < blocks.size(); ++r)
            bytes += pool->conversation_blocks(cid, r).size() * pool->block_bytes(r);
        return bytes;
    }
    strata::core::ConversationId id() const override { return cid; }
    int64_t get_pool_tokens(int64_t tokens) const override { return tokens; }

    /// Makes the model compute this conversation: the seated one's host side goes back to it, this one's into the
    /// model, and the model reads this one's pages.
    void take_seat() {
        if (lead != this) {
            lead->take_seat();
            return;
        }
        if (seated == this)
            return;
        if (seated != nullptr)
            seated->leave_seat();
        m.swap_conversation(state);
        seated = this;
        for (Part* part : peers)
            if (part->is_ready())
                part->point_model();
    }
    /// Takes its host side out of the model when it is seated.
    void leave_seat() {
        if (seated != this)
            return;
        m.swap_conversation(state);
        seated = nullptr;
    }
    /// Gets the tokens its state holds.
    int64_t get_pos() const { return seated == this ? m.pos() : (int64_t) state.tokens.size(); }

    /// Sizes the host and device tables of the growing regions for a conversation of the whole context.
    bool alloc_tables(std::string& err) {
        if (!tables.empty())
            return true;
        const strata::core::StateLayout& layout = pool->layout();
        const strata::core::OnDevice on(dev);
        blocks.resize(layout.regions.size());
        tables.assign(layout.regions.size(), nullptr);
        zeroed.assign(layout.regions.size(), 0);
        for (size_t r = 0; r < layout.regions.size(); ++r) {
            const strata::core::StateRegion& region = layout.regions[r];
            if (region.kind == strata::core::RegionKind::fixed)
                continue;
            const uint64_t rows = region.units_for(m.max_context()), per_block = get_rows_per_block(r);
            const size_t most = (size_t) ((rows + per_block - 1) / per_block + 1);
            blocks[r].reserve(most);
            tables[r] = (uint8_t**) strata::core::gpu_alloc_zeroed(most * sizeof(uint8_t*));
            if (tables[r] == nullptr) {
                for (uint8_t** table : tables)
                    strata::core::gpu_free(table);
                tables.clear();
                err = "no room for the conversation's block tables";
                return false;
            }
        }
        return true;
    }

    /// Writes the blocks the pool gives the conversation into the tables, zeroing the new ones first with
    /// `zero_new` (their pages held other data), and points the model at them when it is seated.
    bool write_tables(std::string& err, bool zero_new = true) {
        const strata::core::OnDevice on(dev);
        const strata::core::StateLayout& layout = pool->layout();
        for (size_t r = 0; r < layout.regions.size(); ++r) {
            if (tables[r] == nullptr)
                continue;
            const std::vector<strata::core::PageId>& runs = pool->conversation_blocks(cid, r);
            blocks[r].clear();
            for (const strata::core::PageId page : runs)
                blocks[r].push_back(pool->page_address(page));
            for (size_t i = zeroed[r]; zero_new && i < blocks[r].size(); ++i)
                if (!strata::core::gpu_memset_async(blocks[r][i], 0, pool->block_bytes(r), nullptr))
                    return fail("zeroing the conversation's new pages", err);
            zeroed[r] = blocks[r].size();
            if (!blocks[r].empty() &&
                !strata::core::gpu_copy(tables[r], blocks[r].data(), blocks[r].size() * sizeof(uint8_t*)))
                return fail("writing the conversation's block tables", err);
        }
        if (seated == lead)
            point_model();
        return true;
    }

    /// Points the model at its fixed region and its rows.
    void point_model() {
        std::vector<KvRows> ckv(at.ckv.size()), ik(at.ik.size());
        for (size_t s = 0; s < at.ckv.size(); ++s) {
            ckv[s] = get_rows(at.ckv[s]);
            ik[s] = get_rows(at.ik[s]);
        }
        m.use_state_memory(stage, get_fixed(at.trunk), get_fixed(at.drafter), at, ckv, ik);
    }

    /// Gets the address of fixed region `r`; null for kNoRegion.
    uint8_t* get_fixed(size_t r) const {
        return r != kNoRegion ? pool->page_address(pool->conversation_blocks(cid, r).at(0)) : nullptr;
    }

    /// Gets how region `r`'s rows lie in its blocks; none for kNoRegion.
    KvRows get_rows(size_t r) const {
        KvRows rows;
        if (r == kNoRegion)
            return rows;
        rows.blocks = tables[r];
        rows.host_blocks = blocks[r].data();
        rows.per_block = (uint32_t) get_rows_per_block(r);
        rows.bytes = (uint32_t) pool->layout().regions[r].unit_bytes;
        return rows;
    }
    uint64_t get_rows_per_block(size_t r) const { return pool->block_bytes(r) / pool->layout().regions[r].unit_bytes; }
    static bool fail(const char* what, std::string& err) {
        err = std::string(what) + " failed";
        return false;
    }
};

/// DeepSeek for the paged engine: the conversations of the batch slots and "main", whose attention reads each row's
/// Part pages; conversation windows share the expert pass; the GPU computes the experts of the card's
/// pool, the CPU the others from the RAM's pool (Options::expert_pages).  A paused conversation's checkpoints are its
/// state beside the pages (save_state), and a prompt that leaves a conversation before its end goes back to one of
/// them (rewind). DSpark guesses (propose); one shared pass verifies each conversation's window (run_rows) and
/// rolls its rejected rows back (keep_rows). The drafter's rings are a fixed region of the conversation,
/// its experts in the GPU's pool before the trunk's (the RAM's pool holds none: it drafts only while all of them are
/// on the GPU). Stages use Model's existing serial or chunked hand-offs; each stage owns a pool, while the lead
/// part owns the conversation's host state.
class PagedDeepSeek final : public strata::serve::PagedModel {
public:
    /// `drafting`: the drafter is switched on at the start; `files`: the experts on the SSD; `ram`: the RAM's pool,
    /// which the model reads (Options::expert_pages).
    PagedDeepSeek(Model& m, const Options& o, const ProgramOptions& x, bool drafting,
                  const std::vector<int>& eos, ExpertFiles& files, strata::core::RamExperts& ram, PagedCards& cards);

    PagedParts make_parts(PagedCard& card, size_t first, size_t count, std::string& err) override;
    bool get_card_layout(const PagedCard& card, int64_t lb, int64_t le, strata::core::StateLayout& layout,
                         std::string& err) const override {
        ConversationRegions at;
        return describe_conversation(m_.geometry(), (int) lb, (int) le,
                                     facts.has_drafter && card.index + 1 == cards_.size(), layout, at, err);
    }
    uint64_t get_drafter_weight_bytes(const PagedCard& card) const override {
        return m_.get_drafter_weight_bytes((int) card.index);
    }
    uint64_t get_gpu_keep(const PagedCard&) const override { return (uint64_t) o_.reserve_mib << 20; }
    std::vector<strata::core::ExpertKey> get_card_ranking(int64_t lb, int64_t le, int64_t& max_experts) const override;
    bool load_expert(const PagedCard& card, strata::core::ExpertKey key, int32_t first, std::string& err) override;
    std::vector<int32_t>& get_residency() override { return residency_; }
    bool publish_residency(const PagedCard& card, std::string& err) override;
    std::vector<float>& get_usage() override { return m_.get_usage(); }
    bool set_card_layers(const PagedCard& card, std::string& err) override {
        return m_.set_stage_layers((int) card.index, (int) card.lb, (int) card.le, err);
    }
    strata::core::LayerSizes get_layer_sizes(uint64_t page_bytes) const override;
    bool can_run_layers(int64_t lb, int64_t le) const override { return lb >= 0 && lb < le && le <= facts.n_layers; }
    strata::core::RegionSpan get_gpu_span(const PagedCard& card) const override {
        return get_conversation_span(m_.geometry(), (int) card.lb, (int) card.le,
                                     facts.has_drafter && card.index + 1 == cards_.size(), false);
    }
    strata::core::RegionSpan get_ram_span(int64_t lb, int64_t le) const override {
        return get_conversation_span(m_.geometry(), (int) lb, (int) le,
                                     facts.has_drafter && le == facts.n_layers, true);
    }
    bool attach_hop(PagedCard& card, std::string& err) override {
        if (facts.cpu_expert_ms == 0) {
            facts.cpu_expert_ms = m_.measure_cpu_expert_ms();
            std::fprintf(stderr, "%s: CPU expert calibration: %.3f ms\n", kProgram.c_str(), facts.cpu_expert_ms);
        }
        return m_.attach_expert_hop((int) card.index, [this, &card](int layer, int expert) {
            const int32_t page = card.lent[(size_t) layer * facts.n_expert + expert];
            return page < 0 ? nullptr : card.experts.pool().page_address(page);
        }, err);
    }
    void set_hop_holders(const int32_t* holders) override { m_.set_hop_holders(holders); }
    void report_hops() const override { m_.report_hops(); }
    bool prepare_images(const strata::serve::RequestLine& request, const std::vector<int64_t>& ids,
                        std::vector<strata::core::ConversationImageKey>& images, std::string& err) override;
    void attach_images(const PagedParts&) override {}
    void start_request(int session, const strata::serve::RequestStart& r) override;
    bool save_state(const PagedParts& parts, std::vector<uint8_t>& state, std::string& err) const override;
    bool restore_state(const std::vector<uint8_t>& state, const PagedParts& parts, const std::vector<int32_t>& live,
                       int64_t kept, std::string& err) const override;
    bool get_session_identity(strata::core::SessionFileIdentity& id, std::string& err) const override {
        std::vector<strata::core::SessionModelFile> files;
        for (const char* name : {"ds4.txt", "tensors.txt", "engram.bin"})
            files.push_back({name, o_.pack + "/" + name, true});
        const std::vector<std::string> paths = m_.get_file_paths();
        for (size_t i = 0; i < paths.size(); ++i)
            files.push_back({"weights " + std::to_string(i), paths[i]});
        if (!strata::core::session_model_fingerprint(files, id.model, err))
            return false;
        strata::core::SessionIdentityBuilder config(0x44533431434f4e46ull);
        config.str("slot identity", m_.slot_identity());
#if defined(STRATA_USE_HIP)
        config.str("backend", "hip");
#else
        config.str("backend", "cuda");
#endif
        config.i64("context", facts.context);
        config.i64("experts per token", m_.get_experts_per_token());
        config.i64("drafter", facts.has_drafter);
        config.i64("drafter taps", facts.has_drafter ? m_.geometry().d_targets.size() : 0);
        id.config = config.digest();
        return true;
    }
    std::vector<int64_t> get_rewind_points(const PagedParts& parts) const override;
    uint64_t get_host_state_bytes(const PagedParts& parts) const override;
    bool rewind(const PagedParts& parts, int64_t tokens, std::string& err) override;
    void forget_drafts(int) override {}
    int64_t get_prompt_chunk() const override { return m_.feed_unit(); }
    bool read_prompt(const PagedParts& parts, const std::vector<int64_t>& ids, int64_t from, int64_t n, bool drafting,
                     const ChunkDone& on_chunk, std::string& err) override;
    bool wait_state(std::string& err) override { return m_.wait_state(err); }
    int get_pipeline_stages() const override {
        // Speculative windows share rollback scratch until the step commits all participants.
        return !facts.has_drafter && cards_.size() > 1 ? (int) cards_.size() : 0;
    }
    bool prepare_group(int group, const std::vector<strata::spec::StepRow>& rows, std::string& err) override;
    bool launch_group(int stage, int group, std::string& err) override {
        return group_rows_[(size_t) group].pending.empty() || m_.launch_group(stage, group, err);
    }
    int poll_group(int stage, int group, std::vector<int32_t>&, std::string& err) override {
        return group_rows_[(size_t) group].pending.empty() ? 1 : m_.poll_group(stage, err);
    }
    bool finish_group(int group, std::vector<int32_t>& picks, std::string& err) override;
    bool keep_group(int, const std::vector<strata::spec::StepRow>&, const std::vector<int>&,
                    bool, std::string&) override { return true; }
    bool run_rows(const std::vector<strata::spec::StepRow>& rows, std::vector<int32_t>& picks,
                  std::string& err) override;
    bool keep_rows(const std::vector<strata::spec::StepRow>& rows, const std::vector<int>& kept, bool,
                   std::string& err) override;
    int propose(strata::serve::PagedPart& last, int session, int32_t token, int64_t pos, int n, int32_t* out,
                std::string& err, strata::spec::DraftContext* draft) override;
    bool has_batched_drafter() const override { return true; }
    bool propose_batch(std::vector<strata::spec::DraftRequest>& requests, std::string& err) override;
    bool switch_drafter(bool on, std::string& err) override;
    void print_info(int slots, int) const override {
        for (const std::string& line : m_.info_lines())
            std::printf("%s\n", line.c_str());
        if (slots > 0)
            std::printf("INFO batch_slots=%d slot_cache=0\n", slots);
    }
    void print_done(const strata::serve::RequestDone& d) const override {
        print_response("DONE %lld %lld %.1f %.1f %s %lld %lld %lld 0 0\n", (long long) d.produced, (long long) d.prompt,
                    d.prompt_ms, d.decode_ms, d.finish, (long long) d.guessed_right, (long long) d.guessed,
                    (long long) d.resumed);
    }
    void end_request() override { m_.save_profile(); }

private:
    /// Gets the experts of trunk layers [lb, le) in the order the GPU's pool takes them (get_ranking), the drafter's
    /// before them while it is on.
    std::vector<strata::core::ExpertKey> get_pool_ranking(int64_t lb, int64_t le) const;
    /// Gets 1 when a proposal can run, 0 when it should be skipped, or -1 on state failure.
    int prepare_draft(Part& part, int session, int64_t pos, int n, std::string& err);
    bool is_drafter_ready() const;
    bool go_to(int64_t pos, std::string& err);
    void end_prompt(Part& p);
    int32_t sample(const Part& p);
    /// Notes the LP line of `pick`, the token after `row`, from the head's logit row `logit_row`
    /// (Model::copy_logits_row; note_logprobs).
    void note_row_logprobs(const strata::spec::StepRow& row, int32_t pick, int logit_row) {
        note_logprobs(row.session, row.pos, pick, m_.vocab(),
                      [&](float* out) { return m_.copy_logits_row(logit_row, out); });
    }
    bool copy_expert(const PagedCard& card, strata::core::ExpertKey key, int32_t first, bool from_ram,
                     std::string& err);

    /// Retains row order and primed picks until the shared scheduler finishes sampling.
    struct GroupRows {
        std::vector<strata::spec::StepRow> rows;
        std::vector<size_t> pending;
        std::vector<int32_t> picks;
    };
    bool prepare_inputs(GroupRows& g, const std::vector<strata::spec::StepRow>& rows,
                        std::vector<int>& tokens, std::string& err);
    void sample_pending_rows(GroupRows& g);
    std::vector<GroupRows> group_rows_;
    SpanCache image_cache_;
    std::vector<Image> pending_images_;
    Model& m_;
    const Options& o_;
    const ProgramOptions x_;
    ExpertFiles& files_;
    strata::core::RamExperts& ram_;
    std::vector<uint8_t> staging_;     ///< an expert read from the files on its way to the GPU
    std::vector<uint64_t> blobs_;      ///< per layer (the trunk's, then the drafter's): one expert's bytes
    std::vector<int32_t> residency_;
    std::vector<Part*> leads_;          ///< The lead stage's parts, by conversation.
    std::vector<PagedCard*> cards_;           ///< Stages in execution order.
    PagedCard* drafter_card_ = nullptr; ///< The last stage, which owns DSpark and its experts.
    Part* seated_ = nullptr;           ///< the conversation the model computes (Part::take_seat), or null
    uint64_t counter_ = 0;             ///< the sampler's draw count, which the old engine keeps across requests
    uint64_t drafter_epoch_ = 0; ///< Increases when DSpark dense weights return.
    bool drafting_ = false;            ///< the drafter is switched on: its experts are wanted in the GPU's pool
    /// the drafter's experts (get_ranking's), which it needs all of on the GPU to draft
    std::vector<strata::core::ExpertKey> drafter_experts_;
    /// Maps each joint window to its conversation and first logit row.
    struct JointWindow { int session, first, count; };
    std::vector<JointWindow> joint_windows_;
    int verified_ = 0; ///< Counts sampled rows in a standalone verify window; joint_windows_ tracks joint passes.
};

/// Gets the Part a PagedPart of this model is.
Part& get_part(strata::serve::PagedPart& part) { return static_cast<Part&>(part); }

strata::core::LayerSizes get_split_sizes(const Geometry& g, const std::vector<uint64_t>& experts, uint64_t page);
int read_model_options(int argc, char** argv, Options& o, ProgramOptions& x, std::string& eos);
std::vector<strata::core::ExpertKey> get_ranking(const Model& m, int64_t lb, int64_t le, bool drafter);
void add_ranks(const Model& m, int64_t lb, int64_t le, std::vector<strata::core::ExpertKey>& ranking);
bool open_card_pool(const Model& m, const PagedSettings& s, const strata::core::StateLayout& layout, uint64_t keep,
                    int64_t max_experts, const std::vector<strata::core::ExpertKey>& ranking, PagedCard& card,
                    std::string& err);

}  // namespace

int run_paged_engine(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // each protocol line reaches the server as it is printed
    strata::platform::fine_timer();
    PagedSettings settings;
    if (const int status = strata::serve::take_paged_options(argc, argv, kProgram, settings); status >= 0)
        return status;
    Options o;
    ProgramOptions x;
    std::string eos = "1";
    if (const int status = read_model_options(argc, argv, o, x, eos); status >= 0)
        return status;
    if (x.layer_split.empty() && o.gpus.size() > 1)
        x.layer_split = "auto";
    if (settings.step.lookahead == 0)   // (no guesses: the drafter is not loaded, as --dspark none)
        o.dspark = false;
    std::string err;
    std::vector<int> stage_devices = o.gpus;
    if (!x.layer_split.empty()) {
        std::vector<int64_t> cuts;
        if (x.layer_split != "auto" && !strata::serve::parse_i64_list(x.layer_split.c_str(), cuts, err)) {
            std::fprintf(stderr, "%s: --layer-split: %s\n", kProgram.c_str(), err.c_str());
            return 2;
        }
        if (!x.split_device.empty()) {
            stage_devices = {o.gpus.empty() ? 0 : o.gpus[0]};
            const std::vector<int> later = parse_ids(x.split_device);
            stage_devices.insert(stage_devices.end(), later.begin(), later.end());
        } else if (stage_devices.size() < 2 && !cuts.empty()) {
            stage_devices = {o.gpus.empty() ? 0 : o.gpus[0]};
            for (size_t i = 0; i < cuts.size(); ++i)
                stage_devices.push_back((int) i + 1);
        }
        if (stage_devices.size() >= 2) {
            const std::string refusal = strata::core::get_hop_refusal(stage_devices);
            if (settings.remote_experts == "on" && !refusal.empty()) {
                std::fprintf(stderr, "%s: --remote-experts on: %s\n", kProgram.c_str(), refusal.c_str());
                return 2;
            }
        } else {
            int count = 0;
            cudaGetDeviceCount(&count);
            stage_devices.clear();
            for (int i = 0; i < count; ++i)
                stage_devices.push_back(i);
        }
        strata::serve::set_paged_runtime_env(settings, strata::core::get_hop_refusal(stage_devices).empty());
        if (stage_devices.size() < 2 || (!cuts.empty() && cuts.size() + 1 != stage_devices.size())) {
            std::fprintf(stderr, "%s: the layer split needs one more stage than cuts\n", kProgram.c_str());
            return 2;
        }
        o.gpus = stage_devices;
        o.tp = 1;
        o.threads = false;
        // Same-GPU stages share an address space; serial windows let cuts move without rebuilding prompt pipelines.
        if (!strata::core::get_hop_refusal(stage_devices).empty())
            o.prefill_chunk = 0;
        o.plan_layers = [&, cuts](const Geometry& g, const std::vector<uint64_t>& blobs,
                                  std::vector<int64_t>& chosen, std::string& why) {
            chosen = cuts;
            if (x.layer_split == "auto") {
                const uint64_t page = strata::core::choose_expert_page_bytes(blobs);
                const auto sizes = get_split_sizes(g, blobs, page);
                std::vector<uint64_t> pools;
                for (size_t i = 0; i < stage_devices.size(); ++i) {
                    const auto copies = std::count(stage_devices.begin(), stage_devices.end(), stage_devices[i]);
                    const int64_t mib = settings.gpu_mib.empty() ? -1 :
                        settings.gpu_mib[std::min(i, settings.gpu_mib.size() - 1)];
                    pools.push_back(mib < 0 ?
                        strata::serve::get_gpu_room(stage_devices[i], (uint64_t) o.reserve_mib << 20, 0) / copies :
                        (uint64_t) mib << 20);
                }
                auto problem = strata::core::make_paged_split_problem(sizes, pools, 1, settings.tokens_min);
                for (size_t i = 0; i < stage_devices.size(); ++i)
                    problem.cards[i].layer_ms = strata::program::split_rules::get_layer_ms(
                        strata::core::get_card_speed(stage_devices[i]));
                const int64_t n = x.gpu_experts < 0 ? (int64_t) g.nl * g.n_exp : x.gpu_experts;
                for (int64_t i = 0; i < n; ++i)
                    problem.ranked_layers.push_back((int32_t) (i % g.nl));
                const auto plan = strata::core::plan_layer_split(problem, "strata-ds41: layer split auto: ");
                strata::core::print_layer_split(plan, problem.ranked_layers.size(), "strata-ds41: layer split auto: ");
                chosen = plan.cuts;
            }
            int64_t previous = 0;
            for (int64_t cut : chosen) {
                if (cut <= previous || cut >= g.nl) {
                    why = "layer cuts must increase strictly inside the trunk";
                    return false;
                }
                previous = cut;
            }
            if (chosen.size() + 1 != stage_devices.size()) {
                why = "the planner found no cut for these stages";
                return false;
            }
            return true;
        };
    } else {
        strata::serve::set_paged_runtime_env(settings, false);
        if (o.gpus.empty())
            o.gpus = {0};
    }
    ExpertFiles files;
    strata::core::RamExperts ram;
    o.expert_pages = &ram;
    Model m;
    if (!m.load(o, err)) {
        std::fprintf(stderr, "%s: %s\n", kProgram.c_str(), err.c_str());
        return 1;
    }
    const std::vector<int> devices = m.get_devices();
    // --dspark-block N is the lookahead's default, else the drafter's block; --dspark off is --drafter off
    const bool has_drafter = m.draft_block() > 0;
    if (settings.step.concurrent < 0)
        settings.step.concurrent = std::max(x.batch, 1);
    if (settings.step.lookahead < 0)
        settings.step.lookahead = !has_drafter ? ((settings.suffix_draft > 0 || settings.lookup_chain > 0) ? 3 : 0) :
                                  o.dspark_block > 0 ? o.dspark_block : m.draft_block();
    if (settings.step.lookahead > 0 && !has_drafter && settings.suffix_draft == 0 && settings.lookup_chain == 0) {
        std::fprintf(stderr, "%s: --lookahead %d needs the DSpark drafter (--dspark on)\n", kProgram.c_str(),
                     settings.step.lookahead);
        return 2;
    }
    if (const std::string why = strata::serve::get_step_settings_problem(settings.step, kWindowRows); !why.empty()) {
        std::fprintf(stderr, "%s: %s\n", kProgram.c_str(), why.c_str());
        return 2;
    }
    settings.drafter = settings.drafter && x.drafting && has_drafter;
    const Geometry& g = m.geometry();
    PagedCards cards;
    const std::vector<int64_t> cuts = m.get_layer_cuts();
    const std::vector<uint64_t> blobs = m.get_expert_bytes();
    strata::core::StateLayout layout;
    ConversationRegions at;
    bool copy_back = false;
    if (!describe_conversation(g, 0, g.nl, has_drafter, layout, at, err) || !m.place_experts(err) ||
        !files.open(m.get_file_paths(), m.get_expert_files(), err)) {
        std::fprintf(stderr, "%s: %s\n", kProgram.c_str(), err.c_str());
        return 1;
    }
    for (size_t i = 0; i < devices.size(); ++i) {
        PagedCard& card = cards.emplace_back();
        card.index = i;
        card.dev = devices[i];
        card.lb = i == 0 ? 0 : cuts[i - 1];
        card.le = i == cuts.size() ? g.nl : cuts[i];
        card.weights_lb = !x.trim_stage_weights ? 0 : card.lb;
        card.weights_le = !x.trim_stage_weights ? g.nl : card.le;
        card.id = "gpu" + std::to_string(card.dev);
        if (std::find(devices.begin(), devices.begin() + i, card.dev) != devices.begin() + i)
            card.id += "." + std::to_string(i);
        strata::serve::measure_card(card, kProgram, *std::max_element(blobs.begin(), blobs.end()));
    }
    if (has_drafter && !settings.drafter && !m.switch_drafter_weights(false, err)) {
        std::fprintf(stderr, "%s: %s\n", kProgram.c_str(), err.c_str());
        return 1;
    }
    PagedDeepSeek model(m, o, x, settings.drafter, parse_ids(eos), files, ram, cards);
    for (PagedCard& card : cards) {
        strata::core::StateLayout stage_layout;
        int64_t most = x.gpu_experts;
        const auto ranking = model.get_card_ranking(card.lb, card.le, most);
        if (!model.get_card_layout(card, card.lb, card.le, stage_layout, err) ||
            !open_card_pool(m, settings, stage_layout, (uint64_t) o.reserve_mib << 20, most, ranking, card, err)) {
            std::fprintf(stderr, "%s: %s\n", kProgram.c_str(), err.c_str());
            return 1;
        }
    }
    PagedCard& card = cards.front();
    if (x.adapt_every > 0 && x.adapt_swaps > 0)   // the routings the engine's adaptive tier counts
        m.get_usage().assign((size_t) g.nl * (size_t) g.n_exp, 0.0f);
    const std::vector<uint64_t> trunk(blobs.begin(), blobs.begin() + g.nl);   // (the RAM's pool: the trunk's experts)
    if (!strata::serve::open_ram_pool(kProgram, settings, files, kRamHeadroom, kCpuLaneExperts, layout, card, trunk,
                                      g.n_exp, get_ranking(m, 0, g.nl, false), card.experts.pool().page_address(0),
                                      0, ram, copy_back, err)) {
        std::fprintf(stderr, "%s: %s\n", kProgram.c_str(), err.c_str());
        return 1;
    }
    const int status = strata::serve::serve_paged_model(model, cards, ram, copy_back, settings);
    m.save_profile();
    return status;
}

namespace {

PagedDeepSeek::PagedDeepSeek(Model& m, const Options& o, const ProgramOptions& x,
                             bool drafting, const std::vector<int>& eos, ExpertFiles& files,
                             strata::core::RamExperts& ram, PagedCards& cards)
    : m_(m), o_(o), x_(x), files_(files), ram_(ram), drafting_(drafting) {
    for (PagedCard& card : cards)
        cards_.push_back(&card);
    drafter_card_ = cards_.back();
    const Geometry& g = m.geometry();
    facts.name = kProgram;
    facts.n_layers = g.nl;
    facts.n_expert = g.n_exp;
    facts.row_bytes = (uint64_t) g.dim * sizeof(float);
    facts.context = o.max_context;
    facts.max_rows = kWindowRows;
    facts.end_tokens.assign(eos.begin(), eos.end());
    facts.sampling.top_k = kDefaultTopK;
    facts.has_drafter = m.draft_block() > 0;
    facts.drafter_label = "DSpark";
    facts.kv_quantization = "FP8 sliding window, FP4 compressed KV, FP4 index";
    facts.drafter_start = "--dspark on and a --lookahead above 0";
    // the engine's tables are the trunk's; the drafter's experts are this model's alone (residency_ has room for them,
    // after the trunk's layers)
    blobs_ = m.get_expert_bytes();
    facts.expert_bytes.assign(blobs_.begin(), blobs_.begin() + g.nl);
    if (facts.has_drafter)
        for (size_t layer = (size_t) g.nl; layer < blobs_.size(); ++layer)
            facts.drafter_expert_bytes += blobs_[layer] * (uint64_t) g.d_n_exp;
    if (cards.size() > 1) {
        facts.split_ranking = get_ranking(m, 0, g.nl, drafting);
        if (x.gpu_experts >= 0 && facts.split_ranking.size() > (size_t) x.gpu_experts)
            facts.split_ranking.resize((size_t) x.gpu_experts);
        facts.split_ranking.erase(std::remove_if(facts.split_ranking.begin(), facts.split_ranking.end(),
            [&](const strata::core::ExpertKey& key) { return key.layer >= g.nl; }), facts.split_ranking.end());
    }
    facts.draft_limits.confidence = o.dspark_conf;
    facts.draft_limits.cumulative = m.get_default_cum();
    facts.draft_limits.max_conversations = facts.has_drafter ? 0 : 1;
    facts.draft_limits.cool_after_rejection = true;
    facts.gpu_experts = x.gpu_experts;
    facts.adapt_every = x.adapt_every;
    facts.adapt_swaps = x.adapt_swaps;
    facts.adapt_min_gain = x.adapt_min_gain;
    facts.adapt_decay = x.adapt_decay;
    staging_.resize(*std::max_element(blobs_.begin(), blobs_.end()));
    residency_.assign(blobs_.size() * (size_t) g.n_exp, strata::core::kNotResident);
    const std::vector<strata::core::ExpertKey> all = get_ranking(m, 0, 0, true);
    drafter_experts_.assign(all.begin(), all.end());
}

std::vector<strata::core::ExpertKey> PagedDeepSeek::get_card_ranking(int64_t lb, int64_t le,
                                                                     int64_t& max_experts) const {
    std::vector<strata::core::ExpertKey> ranking = get_pool_ranking(lb, le);
    max_experts = (int64_t) ranking.size();
    return ranking;
}

std::vector<strata::core::ExpertKey> PagedDeepSeek::get_pool_ranking(int64_t lb, int64_t le) const {
    std::vector<strata::core::ExpertKey> ranking = get_ranking(m_, 0, facts.n_layers, drafting_);
    if (x_.gpu_experts >= 0 && ranking.size() > (size_t) x_.gpu_experts)
        ranking.resize((size_t) x_.gpu_experts);
    ranking.erase(std::remove_if(ranking.begin(), ranking.end(), [&](const strata::core::ExpertKey& key) {
        return key.layer >= facts.n_layers ? le != facts.n_layers : key.layer < lb || key.layer >= le;
    }), ranking.end());
    return ranking;
}

bool PagedDeepSeek::load_expert(const PagedCard& card, strata::core::ExpertKey key, int32_t first, std::string& err) {
    return copy_expert(card, key, first, key.layer < facts.n_layers && ram_.holds(key.layer, key.expert), err);
}

bool PagedDeepSeek::publish_residency(const PagedCard& card, std::string& err) {
    const strata::core::PagePool& pool = card.experts.pool();
    std::vector<const uint8_t*> at(residency_.size(), nullptr);
    for (size_t i = 0; i < residency_.size(); ++i)
        if (const size_t layer = i / (size_t) facts.n_expert;
            residency_[i] != strata::core::kNotResident &&
            (layer >= (size_t) card.lb && layer < (size_t) card.le ||
             layer >= (size_t) facts.n_layers && card.index + 1 == cards_.size()))
            at[i] = pool.page_address((strata::core::PageId) residency_[i]);
    const strata::core::OnDevice on(card.dev);
    return m_.set_expert_addresses(at, err, (int) card.index);
}

/// Copies expert `key` into `card`'s pool from page `first`: from the RAM's pool (`from_ram`, which holds it) or from
/// the files through the staging buffer.
bool PagedDeepSeek::copy_expert(const PagedCard& card, strata::core::ExpertKey key, int32_t first, bool from_ram,
                                std::string& err) {
    const uint8_t* bytes = from_ram ? ram_.blob(key.layer, key.expert)
                                    : files_.copy_blob(key.layer, key.expert, staging_.data()) ? staging_.data()
                                                                                              : nullptr;
    const strata::core::OnDevice on(card.dev);
    if (bytes == nullptr ||
        !strata::core::gpu_copy(card.experts.pool().page_address((strata::core::PageId) first), bytes,
                                (size_t) blobs_[(size_t) key.layer])) {
        err = "copying layer " + std::to_string(key.layer) + " expert " + std::to_string(key.expert) + " to " +
              card.id + " failed";
        return false;
    }
    return true;
}

PagedParts PagedDeepSeek::make_parts(PagedCard& card, size_t first, size_t count, std::string& err) {
    ConversationRegions at;
    strata::core::StateLayout layout;
    const bool drafter = facts.has_drafter && card.index + 1 == cards_.size();
    if (!describe_conversation(m_.geometry(), (int) card.lb, (int) card.le, drafter, layout, at, err))
        return {};
    PagedParts parts;
    for (size_t i = 0; i < count; ++i) {
        parts.push_back(std::make_unique<Part>(m_, at, card.dev, (int) card.index, drafter, seated_));
        Part& part = get_part(*parts.back());
        if (card.index == 0)
            leads_.push_back(&part);
        part.lead = leads_[first + i];
        part.lead->peers.push_back(&part);
    }
    return parts;
}

bool PagedDeepSeek::prepare_images(const strata::serve::RequestLine& request, const std::vector<int64_t>& ids,
                                    std::vector<strata::core::ConversationImageKey>& images, std::string& err) {
    pending_images_.clear();
    if (!request.pictures)
        return true;
    if (m_.image_token() < 0) {
        err = "this engine has no vision tower, so it cannot read pictures";
        return false;
    }
    std::vector<Pic> pictures;
    if (!read_pics(request.file, pictures, err))
        return false;
    int64_t end = 0;
    for (const Pic& picture : pictures) {
        auto rows = image_cache_.get(m_, picture, err);
        if (!rows)
            return false;
        const int64_t n = (int64_t) rows->size() / m_.dim();
        if (picture.start < end || picture.start < 0 || picture.start >= (int64_t) ids.size() ||
            n >= (int64_t) ids.size() - picture.start ||
            !std::all_of(ids.begin() + picture.start, ids.begin() + picture.start + n,
                         [&](int64_t token) { return token == m_.image_token(); })) {
            err = "a picture's span is not a run of image tokens in the prompt";
            return false;
        }
        end = picture.start + n;
        images.push_back({picture.start, picture.hash});
        pending_images_.push_back({picture.start, rows});
    }
    return true;
}

void PagedDeepSeek::start_request(int session, const strata::serve::RequestStart& r) {
    Part& p = *leads_[(size_t) session];
    p.smp = r.sampling;
    p.prompt_open = true;
    p.primed_pos = -1;   // (its pick had the last request's sampling)
    p.pin = 0;
    for (const auto& [key, value] : r.keys)
        if (key == "pin")
            p.pin = std::atoll(value.c_str());
}


// The state beside the pages is the conversation's host side (conversation_state.hpp's bytes).
bool PagedDeepSeek::save_state(const PagedParts& parts, std::vector<uint8_t>& state, std::string&) const {
    Part& p = get_part(*parts[0]);
    p.leave_seat();
    encode_state(p.state, state);
    return true;
}

bool PagedDeepSeek::restore_state(const std::vector<uint8_t>& state, const PagedParts& parts,
                                  const std::vector<int32_t>& live, int64_t, std::string& err) const {
    Part& p = get_part(*parts[0]);
    ConversationState s;
    if (!decode_state(state, s, err))
        return false;
    if (const std::string why = m_.get_state_problem(s); !why.empty()) {
        err = "the paused conversation's checkpoints are damaged: " + why;
        return false;
    }
    // (one more token when the prompt's read took its last one along and the request ended before a step fed it)
    if (s.tokens.size() < live.size() || s.tokens.size() > live.size() + 1 ||
        !std::equal(live.begin(), live.end(), s.tokens.begin())) {
        err = "the paused conversation's checkpoints are another conversation's";
        return false;
    }
    p.leave_seat();
    p.state = std::move(s);
    p.pin_next = false;
    return true;
}

std::vector<int64_t> PagedDeepSeek::get_rewind_points(const PagedParts& parts) const {
    const Part& p = get_part(*parts[0]);
    return !p.is_ready() ? std::vector<int64_t>{} : seated_ == &p ? m_.check_positions() : get_check_positions(p.state);
}

uint64_t PagedDeepSeek::get_host_state_bytes(const PagedParts& parts) const {
    Part& p = get_part(*parts.front());
    p.leave_seat();
    uint64_t bytes = strata::core::get_owned_vector_bytes(p.state.tokens);
    for (const auto& checkpoint : p.state.checks)
        bytes += strata::core::get_owned_vector_bytes(checkpoint.win) +
                 strata::core::get_owned_vector_bytes(checkpoint.comp);
    return bytes;
}

bool PagedDeepSeek::rewind(const PagedParts& parts, int64_t tokens, std::string& err) {
    Part& p = get_part(*parts[0]);
    p.take_seat();
    p.primed_pos = -1;
    p.pin_next = true;
    if (m_.restore(tokens) && m_.pos() == tokens)
        return true;
    err = "no checkpoint at token " + std::to_string(tokens);
    return false;
}

strata::core::LayerSizes PagedDeepSeek::get_layer_sizes(uint64_t page) const {
    return get_split_sizes(m_.geometry(), m_.get_expert_bytes(), page);
}

bool PagedDeepSeek::read_prompt(const PagedParts& parts, const std::vector<int64_t>& ids, int64_t from, int64_t n,
                                bool, const ChunkDone& on_chunk, std::string& err) {
    /// Keeps another conversation's decode positions from selecting this prompt's image rows.
    struct ClearPromptImages {
        Model& model;
        ~ClearPromptImages() { model.set_images({}); }
    } clear{m_};
    m_.set_images(pending_images_);
    Part& p = get_part(*parts[0]);
    p.take_seat();
    if (!go_to(from, err))
        return false;
    // the prompt's last token rides along with the rest, as the old engine reads it (Model::feed_mark: one run of
    // chunks, which gives other numbers than a window of its own), and the token after it is sampled now, for the
    // step that would feed it
    const int64_t end = from + n;
    const bool last = end + 1 == (int64_t) ids.size();
    if (last)
        for (Part* part : p.peers)
            if (!part->reserve(end + 1, err))
                return false;
    // pin=N: a read that crosses N stops there for the shared start's checkpoint, as the old engine's does
    const int64_t pin = p.pin >= kShortestPin && p.pin + 1 < (int64_t) ids.size() ? p.pin : 0;
    int64_t at = from;
    if (pin > from && pin < end) {
        const std::vector<int> start(ids.begin() + from, ids.begin() + pin);
        if (!m_.feed(start.data(), (int) start.size(), 0, err))
            return false;
        m_.checkpoint(kSharedStartCheck);
        at = pin;
    }
    const std::vector<int> tokens(ids.begin() + at, ids.begin() + end + (last ? 1 : 0));
    if (!(last ? m_.feed_mark(tokens.data(), (int) tokens.size(), 1, kBeforeLastCheck, err)
               : m_.feed(tokens.data(), (int) tokens.size(), 0, err)))
        return false;
    if (end == pin)
        m_.checkpoint(kSharedStartCheck);

    if (last) {
        end_prompt(p);
        p.primed_pos = end;
        p.primed_token = tokens.back();
        p.primed_pick = sample(p);
        note_row_logprobs({(int) p.id() - 1, p.primed_token, end}, p.primed_pick, -1);
    }
    return on_chunk(from, n, err);
}

bool PagedDeepSeek::prepare_group(int group, const std::vector<strata::spec::StepRow>& rows, std::string& err) {
    if (group_rows_.size() <= (size_t) group)
        group_rows_.resize((size_t) group + 1);
    GroupRows& g = group_rows_[(size_t) group];
    std::vector<int> tokens;
    if (!prepare_inputs(g, rows, tokens, err))
        return false;
    return tokens.empty() || m_.prepare_group(group, tokens, [this, group](int r) {
        const auto& g = group_rows_[(size_t) group];
        leads_[(size_t) g.rows[g.pending[(size_t) r]].session]->take_seat();
    }, [this, group](int r) {
        const auto& g = group_rows_[(size_t) group];
        for (Part* p : leads_[(size_t) g.rows[g.pending[(size_t) r]].session]->peers)
            p->point_model();
    }, err);
}

bool PagedDeepSeek::finish_group(int group, std::vector<int32_t>& picks, std::string& err) {
    GroupRows& g = group_rows_[(size_t) group];
    if (!g.pending.empty() && !m_.finish_group(group, [this, &g](int r) {
            Part& p = *leads_[(size_t) g.rows[g.pending[(size_t) r]].session];
            p.take_seat();
            for (Part* peer : p.peers)
                peer->point_model();
        }, err))
        return false;
    sample_pending_rows(g);
    picks = g.picks;
    return true;
}

bool PagedDeepSeek::run_rows(const std::vector<strata::spec::StepRow>& rows, std::vector<int32_t>& picks,
                             std::string& err) {
    joint_windows_.clear();
    if (rows.front().session != rows.back().session) {
        verified_ = 0;
        GroupRows g;
        std::vector<int> tokens, counts;
        if (!prepare_inputs(g, rows, tokens, err))
            return false;
        picks = g.picks;
        const auto& pending = g.pending;
        for (size_t first = 0; first < pending.size();) {
            size_t end = first + 1;
            const int session = rows[pending[first]].session;
            while (end < pending.size() && rows[pending[end]].session == session)
                ++end;
            const int count = (int) (end - first);
            joint_windows_.push_back({session, (int) first, count});
            counts.push_back(count);
            first = end;
        }
        if (!pending.empty() && !m_.step_conversations(tokens, [&](int r) {
                leads_[(size_t) rows[pending[(size_t) r]].session]->take_seat();
            }, [&](int r) {
                for (Part* part : leads_[(size_t) rows[pending[(size_t) r]].session]->peers)
                    part->point_model();
            }, counts, err))
            return false;
        for (const JointWindow& window : joint_windows_) {
            Part& p = *leads_[(size_t) window.session];
            p.take_seat();
            if (p.prompt_open)
                end_prompt(p);
            const auto& s = p.smp;
            if (window.count == 1) {
                picks[pending[(size_t) window.first]] = m_.sample_row(window.first, s.temperature, s.top_p,
                    s.top_k, s.min_p, s.seed, counter_++);
                continue;
            }
            std::vector<int> drafts, got;
            for (int i = 1; i < window.count; ++i)
                drafts.push_back(tokens[(size_t) window.first + i]);
            if (!m_.pick_joint_rows(window.session, window.first, drafts, s.temperature, s.top_p, s.top_k,
                                    s.min_p, s.seed, counter_, got, err))
                return false;
            counter_ += got.size();
            for (size_t i = 0; i < got.size(); ++i)
                picks[pending[(size_t) window.first + i]] = got[i];
        }
        for (size_t r = 0; r < pending.size(); ++r)
            note_row_logprobs(rows[pending[r]], picks[pending[r]], (int) r);
        return true;
    }
    const strata::spec::StepRow& row = rows[0];
    Part& p = *leads_[(size_t) row.session];
    verified_ = 0;
    if (rows.size() > 1) {   // A token and its guesses share one verify window.
        std::vector<int> drafts, got;
        for (size_t r = 1; r < rows.size(); ++r)
            drafts.push_back(rows[r].token);
        p.take_seat();
        const strata::serve::Sampling& s = p.smp;
        if (!go_to(row.pos, err) || !m_.verify_rows(row.token, drafts, s.temperature, s.top_p, s.top_k, s.min_p,
                                                    s.seed, counter_, got, err))
            return false;
        std::copy(got.begin(), got.end(), picks.begin());
        verified_ = (int) got.size();
        for (size_t r = 0; r < got.size(); ++r)
            note_row_logprobs(rows[r], got[r], (int) r);
        return true;
    }

    const bool primed = p.primed_pos == row.pos && p.primed_token == row.token && p.get_pos() == row.pos + 1;
    p.primed_pos = -1;
    if (primed) {
        picks[0] = p.primed_pick;
        return true;
    }
    // a prompt the conversation held up to its last token: that token is read here, with the prompt's checkpoints
    const bool closes = p.prompt_open;
    p.take_seat();
    if (!go_to(row.pos, err))
        return false;
    if (closes)
        m_.checkpoint(kBeforeLastCheck);
    if (!m_.step(row.token, true, err))
        return false;
    if (closes)
        end_prompt(p);
    picks[0] = sample(p);
    note_row_logprobs(row, picks[0], -1);
    return true;
}

/// Seats each conversation once, preserving primed picks for synchronous and pipeline passes.
bool PagedDeepSeek::prepare_inputs(GroupRows& g, const std::vector<strata::spec::StepRow>& rows,
                                  std::vector<int>& tokens, std::string& err) {
    g.rows = rows;
    g.pending.clear();
    g.picks.assign(rows.size(), -1);
    for (size_t r = 0; r < rows.size();) {
        size_t end = r + 1;
        while (end < rows.size() && rows[end].session == rows[r].session)
            ++end;
        Part& p = *leads_[(size_t) rows[r].session];
        const bool primed = end == r + 1 && p.primed_pos == rows[r].pos &&
                            p.primed_token == rows[r].token && p.get_pos() == rows[r].pos + 1;
        p.primed_pos = -1;
        if (primed) {
            g.picks[r] = p.primed_pick;
            r = end;
            continue;
        }
        p.take_seat();
        if (!go_to(rows[r].pos, err))
            return false;
        if (p.prompt_open)
            m_.checkpoint(kBeforeLastCheck);
        for (; r < end; ++r) {
            g.pending.push_back(r);
            tokens.push_back(rows[r].token);
        }
    }
    return true;
}

/// Samples in row order after every stage has finished writing retained logits.
void PagedDeepSeek::sample_pending_rows(GroupRows& g) {
    for (size_t r = 0; r < g.pending.size(); ++r) {
        Part& p = *leads_[(size_t) g.rows[g.pending[r]].session];
        p.take_seat();
        if (p.prompt_open)
            end_prompt(p);
        const auto& s = p.smp;
        g.picks[g.pending[r]] = m_.sample_row((int) r, s.temperature, s.top_p, s.top_k, s.min_p, s.seed, counter_++);
        note_row_logprobs(g.rows[g.pending[r]], g.picks[g.pending[r]], (int) r);
    }
}

// A verify window keeps the rows the step accepted; the sampler's draw count moves past every row with a pick, as the
// old engine's does.
bool PagedDeepSeek::keep_rows(const std::vector<strata::spec::StepRow>& rows, const std::vector<int>& kept, bool,
                              std::string& err) {
    if (!joint_windows_.empty()) {
        size_t row = 0, group = 0;
        for (const JointWindow& window : joint_windows_) {
            while (rows[row].session != window.session) {
                const int session = rows[row].session;
                while (row < rows.size() && rows[row].session == session)
                    ++row;
                ++group;
            }
            leads_[(size_t) window.session]->take_seat();
            if (!m_.keep_joint_rows(window.first, window.count, kept[group], err))
                return false;
        }
        return true;
    }
    if (verified_ == 0)
        return true;
    counter_ += (uint64_t) verified_;
    return m_.keep_verified(kept[0], err);
}

// Runs DSpark once its rings and experts are ready; the shared step selects its candidate prefix.
int PagedDeepSeek::propose(strata::serve::PagedPart& last, int session, int32_t token, int64_t pos, int n,
                           int32_t* out, std::string& err, strata::spec::DraftContext* draft) {
    Part& p = *get_part(last).lead;
    const int ready = prepare_draft(p, session, pos, n, err);
    if (ready <= 0)
        return ready;
    std::vector<int> drafts;
    std::vector<float> conf;
    const strata::serve::Sampling& s = p.smp;
    if (!m_.draft(token, drafts, conf, err, s.temperature, s.top_p, s.top_k, s.min_p, s.seed, counter_))
        return -1;
    const int keep = std::min({n, (int) drafts.size(), (int) conf.size()});
    if (!m_.save_joint_draft(session, keep, err))
        return -1;
    std::copy(drafts.begin(), drafts.begin() + keep, out);
    if (draft && draft->probs)
        std::copy_n(conf.begin(), keep, draft->probs);
    for (const strata::core::ExpertKey& key : drafter_experts_)   // (every draft uses its blocks: they stay hot)
        drafter_card_->experts.note_use(key);
    return keep;
}

bool PagedDeepSeek::propose_batch(std::vector<strata::spec::DraftRequest>& requests, std::string& err) {
    std::vector<Model::DraftRequest> batch;
    std::vector<size_t> pending;
    for (size_t i = 0; i < requests.size(); ++i) {
        auto& r = requests[i];
        Part& p = *leads_[(size_t) r.session];
        const int ready = prepare_draft(p, r.session, r.pos, r.limit, err);
        if (ready < 0)
            return false;
        if (ready == 0 || r.pos + m_.draft_block() > facts.context)
            continue;
        const auto& sampling = p.smp;
        batch.push_back({r.session, r.token, r.limit, r.pos, sampling.temperature, sampling.top_p,
                         sampling.min_p, sampling.top_k, sampling.seed, counter_, {}, {}});
        pending.push_back(i);
    }
    if (!m_.draft_conversations(batch, [&](int i) {
            leads_[(size_t) batch[(size_t) i].session]->take_seat();
        }, [&](int i) {
            for (Part* part : leads_[(size_t) batch[(size_t) i].session]->peers)
                part->point_model();
        }, err))
        return false;
    for (size_t i = 0; i < batch.size(); ++i) {
        auto& r = requests[pending[i]];
        const auto& b = batch[i];
        r.count = std::min({r.limit, (int) b.tokens.size(), (int) b.confidence.size()});
        std::copy_n(b.tokens.begin(), r.count, r.tokens.begin());
        std::copy_n(b.confidence.begin(), r.count, r.confidence.begin());
    }
    if (!batch.empty())
        for (const auto& key : drafter_experts_)
            drafter_card_->experts.note_use(key);
    return true;
}

int PagedDeepSeek::prepare_draft(Part& p, int session, int64_t pos, int n, std::string& err) {
    if (!m_.save_joint_draft(session, 0, err))
        return -1;
    if (n <= 0 || !drafting_ || p.prompt_open || p.primed_pos == pos ||
        !is_drafter_ready())
        return 0;
    p.take_seat();
    if (p.drafter_epoch != drafter_epoch_) {
        if (!m_.reseed_drafter(pos, err))
            return -1;
        p.drafter_epoch = drafter_epoch_;
    }
    return go_to(pos, err) ? 1 : -1;
}

bool PagedDeepSeek::switch_drafter(bool on, std::string& err) {
    if (!m_.switch_drafter_weights(on, err))
        return false;
    if (on && !drafting_)
        ++drafter_epoch_;
    drafting_ = on;
    for (PagedCard* card : cards_) {
        const auto ranking = get_pool_ranking(card->lb, card->le);
        int64_t most = (int64_t) ranking.size();
        if (on) {
            const auto report = card->experts.pool().report();
            uint64_t room = report.experts.bytes + report.free.bytes;
            int64_t fit = 0;
            for (const strata::core::ExpertKey& key : ranking) {
                const uint64_t page = card->experts.pool().page_bytes();
                const uint64_t bytes = (blobs_[(size_t) key.layer] + page - 1) / page * page;
                if (bytes > room)
                    break;
                room -= bytes;
                ++fit;
            }
            most = std::min(most, fit);
        }
        card->experts.set_ranking(ranking, most);
        if (!publish_residency(*card, err))
            return false;
    }
    return true;
}

/// Tells whether the GPU's pool holds every one of the drafter's experts, which its blocks need.
bool PagedDeepSeek::is_drafter_ready() const {
    return std::all_of(drafter_experts_.begin(), drafter_experts_.end(), [&](const strata::core::ExpertKey& k) {
        return residency_[(size_t) k.layer * (size_t) facts.n_expert + (size_t) k.expert] >= 0;
    });
}

/// Keeps the model's state at the end of `p`'s prompt as a checkpoint, and the one before its last token as the anchor
/// when its conversation started over or went back (Part::pin_next), as the old engine does.
void PagedDeepSeek::end_prompt(Part& p) {
    m_.checkpoint(kPromptEndCheck);
    if (p.pin_next)
        m_.pin();
    p.pin_next = p.prompt_open = false;
}

/// Gets the token after the model's last logits by `p`'s request's sampling.
int32_t PagedDeepSeek::sample(const Part& p) {
    const strata::serve::Sampling& s = p.smp;
    return m_.sample(s.temperature, s.top_p, s.top_k, s.min_p, s.seed, counter_++);
}

/// Makes the model's state the seated conversation's at `pos` tokens: it is there, or one past it when the prompt's
/// read took the next token along and no step fed it (the request ended first), and goes back to the checkpoint
/// before it.
bool PagedDeepSeek::go_to(int64_t pos, std::string& err) {
    if (m_.pos() == pos)
        return true;
    if (m_.pos() == pos + 1 && m_.restore(pos) && m_.pos() == pos)
        return true;
    err = "the model's state is at token " + std::to_string(m_.pos()) + ", the conversation's at " +
          std::to_string(pos);
    return false;
}

/// Reads the model's own options, strata-ds41's (`x`) and --eos from `argv` (the paged engine's are gone) into `o`, `x`
/// and `eos`.  -1: go on, else the exit status.
int read_model_options(int argc, char** argv, Options& o, ProgramOptions& x, std::string& eos) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto val = [&](const char* what) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s: %s needs a value\n", kProgram.c_str(), what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (read_option(a, val, o, x.drafting))
            continue;
        if (a == "--eos")
            eos = val("--eos");
        else if (a == "--batch")
            x.batch = std::max(0, std::atoi(val("--batch").c_str()));
        else if (a == "--expert-cache") {
            const std::string v = val("--expert-cache");
            x.gpu_experts = v == "auto" ? -1 : std::atoll(v.c_str());
        } else if (a == "--adapt-every")
            x.adapt_every = std::atoi(val("--adapt-every").c_str());
        else if (a == "--adapt-swaps")
            x.adapt_swaps = std::atoi(val("--adapt-swaps").c_str());
        else if (a == "--adapt-min-gain")
            x.adapt_min_gain = (float) std::atof(val("--adapt-min-gain").c_str());
        else if (a == "--adapt-decay")
            x.adapt_decay = (float) std::atof(val("--adapt-decay").c_str());
        else if (a == "--layer-split")
            x.layer_split = val("--layer-split");
        else if (a == "--split-device")
            x.split_device = val("--split-device");
        else if (a == "--trim-stage-weights")
            x.trim_stage_weights = true;
        else if (a != "--serve") {
            std::fprintf(stderr, "%s: unknown option %s\n", kProgram.c_str(), a.c_str());
            return 2;
        }
    }
    if (o.pack.empty()) {
        std::fprintf(stderr, "%s: --pack <model>/maya-ds4 is needed\n", kProgram.c_str());
        return 2;
    }
    if (o.profile.empty())
        o.profile = o.pack + "/expert-profile.bin";   // "none": no profile
    if (o.profile == "none")
        o.profile.clear();
    if (!o.expert_cache)   // --no-expert-cache: the experts stay where the start put them, as strata-dsv41's do
        x.adapt_swaps = 0;
    return -1;

}

/// Gets the experts of trunk layers [lb, le) in the order the GPU's pool takes them: each layer's hottest (Model::
/// get_heat_order), then each layer's second, and so on, so that a pool holds the same share of every layer, as the
/// old engine's VRAM does; with `drafter`, the DSpark blocks' experts (the layers after the trunk's) first, in the same
/// order, as the old engine places them before the trunk's.
std::vector<strata::core::ExpertKey> get_ranking(const Model& m, int64_t lb, int64_t le, bool drafter) {
    std::vector<strata::core::ExpertKey> ranking;
    const int nl = m.geometry().nl;
    if (drafter)
        add_ranks(m, nl, nl + m.geometry().n_mtp, ranking);
    add_ranks(m, lb, le, ranking);
    return ranking;
}

/// Adds the experts of layers [lb, le) to `ranking` rank by rank (get_ranking's order).
void add_ranks(const Model& m, int64_t lb, int64_t le, std::vector<strata::core::ExpertKey>& ranking) {
    std::vector<std::vector<int>> order;
    size_t most = 0;
    for (int64_t l = lb; l < le; ++l)
        most = std::max(most, order.emplace_back(m.get_heat_order((int) l)).size());
    for (size_t rank = 0; rank < most; ++rank)
        for (int64_t l = lb; l < le; ++l)
            if (rank < order[(size_t) (l - lb)].size())
                ranking.push_back({(int32_t) l, order[(size_t) (l - lb)][rank]});
}

/// Opens `card`'s page pool for conversations of `layout` and the first `max_experts` experts of `ranking` (-1: every
/// one) in the GPU memory the model left: --gpu-mib, by default what they and one conversation of the whole context
/// take, leaving `keep` free.  Its page is the trunk's experts' (the drafter's take whole pages, a few bytes unused).
bool open_card_pool(const Model& m, const PagedSettings& s, const strata::core::StateLayout& layout, uint64_t keep,
                    int64_t max_experts, const std::vector<strata::core::ExpertKey>& ranking, PagedCard& card,
                    std::string& err) {
    const std::vector<uint64_t> blobs = m.get_expert_bytes();
    const uint64_t page =
        strata::core::choose_expert_page_bytes({blobs.begin(), blobs.begin() + m.geometry().nl});
    if (max_experts < 0)
        max_experts = (int64_t) ranking.size();
    uint64_t most = strata::serve::get_conversation_bytes(layout, m.max_context(), page), room = 0;
    for (size_t i = 0; i < ranking.size() && (int64_t) i < max_experts; ++i)
        most += (blobs[(size_t) ranking[i].layer] + page - 1) / page * page;
    const int64_t mib = s.gpu_mib.empty() ? -1 : s.gpu_mib[std::min(card.index, s.gpu_mib.size() - 1)];
    const uint64_t bytes = strata::serve::get_gpu_pool_bytes(card.dev, keep, mib, most, room);
    strata::core::PagePoolConfig config;
    config.page_bytes = page;
    config.chunk_bytes = strata::serve::get_chunk_bytes(bytes, page);
    config.tokens_min = std::clamp<int64_t>(s.tokens_min, 0, m.max_context());
    config.label = "Expert and conversation pool";
    config.label_expert = [layers = m.geometry().nl](strata::core::ExpertKey key, uint64_t bytes) {
        return strata::core::MemoryLabel{"experts", "", (key.layer >= layers ? "Drafter / " : "") +
            strata::core::get_expert_memory_label(key.layer, bytes)};
    };
    const strata::core::OnDevice on(card.dev);
    if (!card.experts.open(strata::serve::get_gpu_chunk_memory(card.dev, config.chunk_bytes), bytes, layout, config,
                           blobs,
                           m.geometry().n_exp, max_experts, ranking, err))
        return false;
    card.experts.set_ranking(ranking, max_experts);   // (opened without a cache, it wants none of them until then)
    strata::serve::say_gpu_pool(kProgram, card, room, config.chunk_bytes);
    return true;
}

/// Prices DeepSeek's varying expert blobs and compressed state in the shared split planner.
strata::core::LayerSizes get_split_sizes(const Geometry& g, const std::vector<uint64_t>& experts, uint64_t page) {
    std::vector<KvSource> sources;
    std::vector<int> source_of;
    std::string err;
    (void) find_kv_sources(g, sources, source_of, err);
    strata::core::LayerSizes sizes;
    for (int il = 0; il < g.nl; ++il) {
        sizes.dense.push_back(0);   // (outside the pools)
        sizes.expert.push_back((experts[(size_t) il] + page - 1) / page * page);
        sizes.fixed_state.push_back((uint64_t) g.win * kv_row_bytes(kKvFp8, g.hd));
        sizes.token_bytes.push_back(0);
    }
    for (const KvSource& s : sources) {   // a source's rows, counted at its layer
        sizes.fixed_state[(size_t) s.layer] += s.ratio > 1 ? 2 * (uint64_t) s.ratio * g.hd * sizeof(float) : 0;
        sizes.token_bytes[(size_t) s.layer] = (kv_row_bytes(kKvFp4, g.hd) + kv_row_bytes(kKvFp4p, g.ix_hd)) / s.ratio;
    }
    return sizes;
}

}  // namespace

}  // namespace strata::ds4
