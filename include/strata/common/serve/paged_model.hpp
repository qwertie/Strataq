// include/strata/common/serve/paged_model.hpp - what the paged engine (paged_engine.hpp) asks of a model: its cards'
// state layouts and experts, a conversation's state on each card, reading a prompt, the step and the drafter
// (docs/MEMORY_MANAGER_PLAN.md, "One shared paged engine").  Qwen3.8 (src/models/qwen38/paged_model.cpp) and DeepSeek
// V4.1 (src/models/deepseekv41/paged_model.cpp) implement it.
//
// The model is a class with virtual functions, as DraftSource (strata/common/spec/draft_source.hpp) is, rather than a
// struct of functions as spec::StepModel and EngineHooks are: the engine also keeps objects the model makes (a
// conversation's state on a card, PagedPart), and the model has some thirty operations.
#pragma once

#include "strata/base/gpu/links.hpp"
#include "strata/common/experts/expert_pages.hpp"
#include "strata/common/experts/layer_split.hpp"
#include "strata/common/serve/request.hpp"
#include "strata/common/spec/step.hpp"
#include "strata/common/state/conversation_file.hpp"
#include "strata/common/state/pool_copy.hpp"

#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace strata::serve {

/// One card of a layer split, or the GPU without one: the layers it runs and its page pool, which holds the experts
/// of its layers and the conversations' state for them.  The model's start makes the cards, in the order of their
/// layers, fills every field but `lent` and opens `experts`; the engine then sets lb and le when the layer cut moves,
/// and `lent` when the cards lend pages.  The model keeps its own side of each card (its weights, prompt path and
/// verify windows) by `index`.
struct PagedCard {
    size_t index = 0;                             ///< its place among the cards
    int dev = 0;
    int64_t lb = 0, le = 0;                       ///< it runs layers [lb, le)
    /// its memory's id in GET_MEMORY_USE and CHANGE_SETTINGS: gpu<dev>, and gpu<dev>.<index> for a card on the GPU of
    /// an earlier card (the test of a split on one GPU)
    std::string id;
    /// Bounds the cut to layers whose dense tensors the model can address or reload on this card.
    /// --trim-stage-weights restricts this to the initial interval.
    int64_t weights_lb = 0, weights_le = 0;
    strata::core::LinkSpeed link;                 ///< its link to RAM, measured at start
    double layer_ms = 0;                          ///< the predicted ms of one layer of a decode window on it
    strata::core::ExpertPages experts;            ///< the page pool: its experts and the conversations' pages
    /// the most experts one refill puts into its pool (strata::core::get_refill_batch)
    int refill_batch = 0;
    /// per expert (layer * n_expert + expert) of another card's layers that its pool holds: its slot, else
    /// kNotResident (empty unless the cards lend pages)
    std::vector<int32_t> lent;
};
using PagedCards = std::deque<PagedCard>;   ///< (a deque: a card stays where it is as the next one is added)

/// A conversation's state on one card, which the model makes (PagedModel::make_parts): what the model keeps of it
/// there and the pages of the card's pool that hold its growing state.
class PagedPart {
public:
    virtual ~PagedPart() = default;

    /// Tells whether it has its fixed regions in the card's pool (init).
    virtual bool is_ready() const = 0;
    /// Gives it its fixed regions in `pool` as conversation `id` there; false (and `err`) when the pool has no room.
    virtual bool init(strata::core::PagePool& pool, strata::core::ConversationId id, std::string& err) = 0;
    /// Gives it the pages for its first `tokens` tokens; false (and `err`) when the pool has no room.
    virtual bool reserve(int64_t tokens, std::string& err) = 0;
    /// Tells whether reserve(tokens) would leave the pages and device tables untouched.
    virtual bool can_reserve_without_growth(int64_t) const { return false; }
    /// Gives back its growing pages beyond its first `tokens` tokens.
    virtual void truncate(int64_t tokens) = 0;
    /// Gives back every page it holds, its fixed regions too (it is not ready after).
    virtual void release() = 0;
    /// Points the model at its pages again after a pool shrink moved some of them.
    virtual bool refresh(std::string& err) = 0;
    /// Clears the model's state of it, so that a new conversation starts there.
    virtual void clear_state() = 0;
    /// Makes it the state of layers [lb, le), as its card runs after the layer cut moved; it holds no pages then.
    virtual void set_layers(int64_t lb, int64_t le) = 0;
    /// Gets the bytes of the pool pages it holds.
    virtual uint64_t bytes() const = 0;
    /// Gets its conversation id in the card's pool (given to init).
    virtual strata::core::ConversationId id() const = 0;
    /// Gets the tokens whose pages a pool of its layout keeps for a conversation of `tokens` tokens (whole blocks).
    virtual int64_t get_pool_tokens(int64_t tokens) const = 0;
};
using PagedParts = std::vector<std::unique_ptr<PagedPart>>;   ///< a conversation's state on each card, in their order

/// What the engine knows of a model that does not change while it runs.
struct PagedModelFacts {
    /// the program's name ("strata-q38"): the start of its stderr lines and of some ERR lines, and the name of the
    /// temporary folder's subfolder for paused conversations
    std::string name;
    int64_t n_layers = 0, n_expert = 0;
    int64_t context = 0;                    ///< the most tokens a conversation holds (--max-context)
    int max_rows = 0;                       ///< the most rows one step may run (its verify windows' room)
    std::vector<int64_t> end_tokens;        ///< the tokens that end an answer
    Sampling sampling;                      ///< a request's sampling before its keys (the old engine's defaults)
    /// a drafter guesses tokens (propose), its state in the conversations' pages; without one the engine never calls
    /// forget_drafts, propose or switch_drafter
    strata::spec::DraftLimits draft_limits; ///< Defaults for the shared confidence and scheduling rules.
    bool has_drafter = false;
    /// what the start needs for a drafter, for the refusal of `drafter on` without one ("--mtp and ...")
    std::string drafter_start;
    /// Names the loaded prediction model for the Memory settings UI.
    std::string drafter_label = "model";
    /// Describes the formats used for token state in the shared memory report.
    std::string kv_quantization;
    /// Accounts for prediction experts in addition to reloadable dense weights.
    uint64_t drafter_expert_bytes = 0;
    /// the most conversations one step may run; 0: as many as the step has rows
    int max_concurrent = 0;
    bool serial_conversations = false;     ///< the model switches shared inputs between conversations
    uint64_t extra_resident_vram = UINT64_MAX; ///< Smallest card VRAM allowing an extra cached conversation by default.
    std::vector<uint64_t> expert_bytes;     ///< per layer: one expert's bytes
    /// on a layer split, the first --expert-cache experts of the whole model's profile, hottest first (what one GPU
    /// would hold); empty without a split
    std::vector<strata::core::ExpertKey> split_ranking;
    int64_t gpu_experts = 0;                ///< the most experts the GPUs' pools hold (--expert-cache)
    /// the adaptive tier (the old engine's --adapt-every, --adapt-swaps, --adapt-min-gain and --adapt-decay, a factor
    /// per 4 decode steps)
    int adapt_every = 4, adapt_swaps = 0;
    float adapt_min_gain = 0, adapt_decay = 1;
    uint64_t row_bytes = 0;                 ///< the bytes of one row's hidden state, which a hop carries each way
    /// the ms the CPU takes for one routed expert of a decode step, against which a hop to another card is weighed
    double cpu_expert_ms = 0;
};

/// The memory a prompt path borrows from the pools while it reads a prompt (PagedModel::get_prompt_buffers).
struct PromptBuffers {
    uint64_t device = 0;   ///< bytes from the pool of the card it runs on
    uint64_t host = 0;     ///< bytes from the RAM's pool
};

/// A request as its conversation's model sees it (PagedModel::start_request).
struct RequestStart {
    Sampling sampling;
    /// the request line's keys, in its order: the sampling keys and the engine's own (priority) among them
    std::vector<std::pair<std::string, std::string>> keys;
};


/// What a request's DONE line reports (PagedModel::print_done).
struct RequestDone {
    int64_t produced = 0;                ///< the tokens answered
    int64_t prompt = 0;                  ///< the prompt's tokens
    int64_t resumed = 0;                 ///< of them, the tokens the conversation held already
    int64_t read = 0;                    ///< of them, the tokens read (the rest of the prompt but its last token)
    double prompt_ms = 0, decode_ms = 0;
    const char* finish = "length";       ///< why it ended: stop, length, cancel, ...
    int64_t guessed = 0, guessed_right = 0;   ///< the drafter's guesses, and how many were kept
};

/// A model that the paged engine runs: the engine keeps the conversations, the pools and the protocol, the model
/// computes.  A function that takes a PagedCard gets one of the engine's cards; one that takes a session gets a
/// conversation's index (Held::session in the engine), from 0 to the batch slots.
class PagedModel {
public:
    virtual ~PagedModel() = default;

    /// What the engine reads of the model; the model fills it before serve_paged_model.
    PagedModelFacts facts;
    /// Per session: the LP lines its request asked for, which the engine starts with each request and prints with
    /// the tokens a step keeps; the model notes them as it picks tokens (note_logprobs).
    std::vector<RowLogprobs> logprobs;

    /// Notes the LP line of `token`, picked after `session`'s row at `pos`, when its request asked for one; `copy`
    /// then writes that row's head logits (`vocab` floats) on the host.
    void note_logprobs(int session, int64_t pos, int32_t token, int64_t vocab,
                       const std::function<bool(float*)>& copy) {
        if ((size_t) session < logprobs.size())
            logprobs[(size_t) session].note(pos, token, vocab, copy);
    }

    /// Validates a request's pictures before taking a conversation, and fills the empty `images` with where each
    /// picture starts and a hash of its content, in prompt order.
    virtual bool prepare_images(const RequestLine& request, const std::vector<int64_t>& ids,
                                std::vector<strata::core::ConversationImageKey>& images, std::string& err) = 0;
    /// Attaches the validated pictures to the selected conversation.
    virtual void attach_images(const PagedParts& parts) = 0;

    // ---- per card

    /// Creates state for sessions [first, first + count) on one card during admission.
    /// Returns empty with `err` when the card cannot allocate its working buffers.
    virtual PagedParts make_parts(PagedCard& card, size_t first, size_t count, std::string& err) = 0;
    /// Gets the regions a conversation keeps in `card`'s pool when the card runs layers [lb, le).
    virtual bool get_card_layout(const PagedCard& card, int64_t lb, int64_t le, strata::core::StateLayout& layout,
                                 std::string& err) const = 0;
    /// Gets the GPU memory `card`'s pool must leave free besides what is allocated there now: the model's reserve
    /// and its buffers there that are not allocated (a drafter switched off).  The engine weighs it against the GPU's
    /// free memory when a pool grows.
    virtual uint64_t get_gpu_keep(const PagedCard& card) const = 0;
    /// Gets reloadable dense bytes that trade space with the card's pool when drafting changes.
    virtual uint64_t get_drafter_weight_bytes(const PagedCard&) const { return 0; }
    /// Gets the experts of the profile of layers [lb, le), hottest first; on a layer split, `max_experts` (the most
    /// the GPUs hold) becomes how many of them are among the whole profile's first `max_experts`.
    virtual std::vector<strata::core::ExpertKey> get_card_ranking(int64_t lb, int64_t le,
                                                                  int64_t& max_experts) const = 0;
    /// Copies expert `key`'s bytes into `card`'s pool from page `first`: the card's ExpertPages calls it when it takes
    /// an expert in (ExpertPages::Hooks::load).
    virtual bool load_expert(const PagedCard& card, strata::core::ExpertKey key, int32_t first, std::string& err) = 0;
    /// Gets the residency table: per expert (layer * n_expert + expert), its first page (ExpertPages' slot) in the pool
    /// of the card that runs its layer, else kNotResident.  The engine writes it as experts come and go and then calls
    /// publish_residency for each card; it reads it to choose the adaptive tier's swaps.
    virtual std::vector<int32_t>& get_residency() = 0;
    /// Tells `card`'s kernels where its layers' experts are now, in the model's own form: Qwen3.8 copies the table to
    /// the card's device (its kernels take an expert at its pool's arena plus its slot); DeepSeek's kernels take a
    /// table of addresses per layer (PagePool::page_address of each first page).
    virtual bool publish_residency(const PagedCard& card, std::string& err) = 0;
    /// Gets the routings each expert (layer * n_expert + expert) of the trunk's PagedModelFacts::n_layers layers had
    /// in the decode steps, which the adaptive tier counts and decays; empty when the tier is off.  A card's ranking
    /// may hold experts past them (a drafter's), which have no count.
    virtual std::vector<float>& get_usage() = 0;
    /// Makes the model run layers [card.lb, card.le) on `card` (the layer cut moved); false (and `err`) when its
    /// dense tensors could not reload or its windows could not discard the old layers. The engine calls it
    /// with the card's new range while every conversation is paused, before updating its parts' ranges.
    virtual bool set_card_layers(const PagedCard& card, std::string& err) = 0;
    /// Gets each layer's sizes for the layer split's planner, an expert in pages of `page_bytes`.
    virtual strata::core::LayerSizes get_layer_sizes(uint64_t page_bytes) const = 0;
    /// Tells whether one card can run layers [lb, le) (the model's own rule: Qwen3.8 needs both layer kinds).
    virtual bool can_run_layers(int64_t lb, int64_t le) const = 0;
    /// Gets the regions of `card`'s pool a paused conversation's copy takes.
    virtual strata::core::RegionSpan get_gpu_span(const PagedCard& card) const = 0;
    /// Gets where the part of a conversation of a card that runs layers [lb, le) is in the RAM pool's layout, which
    /// has the whole model's regions.
    virtual strata::core::RegionSpan get_ram_span(int64_t lb, int64_t le) const = 0;
    /// Makes `card` able to compute the experts of other cards' layers that its pool holds (PagedCard::lent, already
    /// sized).  The engine calls it as it starts, for each card, when there are several cards, a split_ranking and
    /// --remote-experts is not off.
    virtual bool attach_hop(PagedCard& card, std::string& err) = 0;
    /// Gives the model the table, per layer, of the card whose pool holds that layer's lent experts (-1: none); it
    /// stays where it is while the engine runs.
    virtual void set_hop_holders(const int32_t* holder) = 0;
    /// Says on stderr what the cards computed in hops so far.
    virtual void report_hops() const = 0;

    // ---- per conversation

    /// Tells the model of `session`'s next request: its sampling and the keys of the model's own.
    virtual void start_request(int session, const RequestStart& r) = 0;
    /// Copies the state the model keeps beside a conversation's pages on each card (`parts`) to `state`, in a form
    /// that does not depend on the layer cut.  The engine calls it as it pauses the conversation, before its pages
    /// leave the cards' pools.
    virtual bool save_state(const PagedParts& parts, std::vector<uint8_t>& state, std::string& err) const = 0;
    /// Puts `state` (save_state's) back into `parts`, each card its layers', which may be another cut's than when it
    /// was saved; `live` are the conversation's tokens, of which its pages hold the first `kept`.  With fewer than all
    /// of them, a rewind to `kept` follows, so the state that belongs after all of `live` need not be put back.  The
    /// engine calls it as it resumes the conversation, after its pages are back.
    virtual bool restore_state(const std::vector<uint8_t>& state, const PagedParts& parts,
                               const std::vector<int32_t>& live, int64_t kept, std::string& err) const = 0;
    /// Gets what a conversation saved to a file (a pause file on the SSD, a saved session) is bound to into `id`: a
    /// fingerprint of the model's files, and one of the settings that change what the saved bytes mean.  False (and
    /// `err`) when a model file cannot be read.  The engine asks once, when it first needs an identity.
    virtual bool get_session_identity(strata::core::SessionFileIdentity& id, std::string& err) const = 0;
    /// Tells whether a rewind preserves the later branch as a paused conversation.
    virtual bool keeps_rewind_branch(const PagedParts&, int64_t) const { return true; }
    /// Gets the positions short of its end that the conversation of `parts` can go back to (rewind), so that a prompt
    /// that leaves it there continues it (strata/common/state/resume_point.hpp); none by default.
    virtual std::vector<int64_t> get_rewind_points(const PagedParts&) const { return {}; }
    /// Gets heap capacity owned by this conversation's host-side checkpoints and metadata, excluding page pools.
    virtual uint64_t get_host_state_bytes(const PagedParts&) const { return 0; }
    /// Makes the conversation of `parts` hold its first `tokens` tokens, one of get_rewind_points; false (and `err`)
    /// when it cannot.  The engine then gives back its pages beyond them (PagedPart::truncate).
    virtual bool rewind(const PagedParts&, int64_t tokens, std::string& err) {
        err = "the conversation cannot go back to token " + std::to_string(tokens);
        return false;
    }
    /// Forgets what the drafter knows of `session`: its conversation starts a request or the drafter came back.
    virtual void forget_drafts(int session) = 0;

    // ---- the prompt, the step and the drafter

    /// Gets the maximum tokens per prompt piece between inspection and concurrent decode opportunities.
    virtual int64_t get_prompt_chunk() const = 0;
    /// Gets the largest chunk a prompt is read in (--prefill: a ceiling), or 0 when the prompt path keeps buffers of
    /// its own.  The engine reads each prompt in the largest chunk up to it whose buffers its pools can lend then.
    virtual int64_t get_prompt_ceiling() const { return 0; }
    /// Gets what `card`'s prompt path borrows to read `tokens` tokens in chunks of `chunk`; nothing when such a read
    /// needs no buffers (Qwen3.8 reads a short prompt through its verify windows).
    virtual PromptBuffers get_prompt_buffers(const PagedCard&, int64_t /*chunk*/, int64_t /*tokens*/) const {
        return {};
    }
    /// Lays `card`'s prompt path out for chunks of `chunk` tokens in the lent `device` and `host` memory
    /// (get_prompt_buffers' bytes; null where none); with `chunk` 0, waits until the path no longer reads them, as
    /// they go back.  False (and `err`) when it cannot.
    virtual bool lay_out_prompt(const PagedCard&, int64_t /*chunk*/, uint8_t* /*device*/, uint8_t* /*host*/,
                                std::string&) {
        return true;
    }
    /// Called after each chunk of a prompt read, `n` tokens from `pos0`; false (and `err`) stops the read.
    using ChunkDone = std::function<bool(int64_t pos0, int64_t n, std::string& err)>;
    /// Reads `ids[from, from + n)` into the conversation of `parts`, calling `on_chunk` after each chunk; the drafter
    /// reads it too when `drafting`.  `ids` is the whole prompt; the engine may read it in pieces of
    /// get_prompt_chunk tokens with inspections and decode steps of other conversations between them.
    virtual bool read_prompt(const PagedParts& parts, const std::vector<int64_t>& ids, int64_t from, int64_t n,
                             bool drafting, const ChunkDone& on_chunk, std::string& err) = 0;
    /// Supplies the consumed tokens for each row's conversation, valid until run_rows returns.
    virtual void set_step_histories(const std::vector<const std::vector<int32_t>*>&) {}
    /// Runs a step's rows (spec::StepModel::run); picks[r] gets row r's next token, or -1 when its logits were not
    /// finite.
    virtual bool run_rows(const std::vector<strata::spec::StepRow>& rows, std::vector<int32_t>& picks,
                          std::string& err) = 0;
    /// Gets the independently driven stage count; zero leaves pipeline groups unavailable.
    virtual int get_pipeline_stages() const { return 0; }
    /// Captures a group's graphs and stores its immutable row inputs before any stage launches.
    virtual bool prepare_group(int, const std::vector<strata::spec::StepRow>&, std::string&) { return false; }
    /// Starts one prepared group on an idle stage without waiting for its GPU.
    virtual bool launch_group(int, int, std::string&) { return false; }
    /// Services a running stage; returns 1 when complete, 0 while busy, or -1 on error.
    virtual int poll_group(int, int, std::vector<int32_t>&, std::string&) { return -1; }
    /// Finalizes a group's picks after all stages drain; models may defer sampling until this point.
    virtual bool finish_group(int, std::vector<int32_t>&, std::string&) { return true; }
    /// Commits one group's rows after every stage has drained.
    virtual bool keep_group(int, const std::vector<strata::spec::StepRow>&, const std::vector<int>&,
                            bool, std::string&) { return false; }
    /// Keeps the first kept[c] of the c-th conversation's rows of the last run (spec::StepModel::keep); the drafter
    /// reads them when `drafting`.
    virtual bool keep_rows(const std::vector<strata::spec::StepRow>& rows, const std::vector<int>& kept,
                           bool drafting, std::string& err) = 0;
    /// Prepares model-specific drafting for the conversations sharing the next pass.
    virtual void begin_step(int) {}
    /// Completes outstanding state writes before commands, page reuse or a completed request can expose them.
    virtual bool wait_state(std::string&) { return true; }
    /// Writes up to `n` guesses of the tokens after `token` at `pos` in `session` to `out` from its state on the last
    /// card (`last`); gets how many, or -1 (and `err`) when the drafter failed.  It is called before each of the
    /// conversation's steps (spec::StepModel::propose), also with `n` 0. `draft` supplies the shared window policy
    /// and receives the proposal's confidences.
    virtual int propose(PagedPart& last, int session, int32_t token, int64_t pos, int n, int32_t* out,
                        std::string& err, strata::spec::DraftContext* draft) = 0;
    /// Loads (`on`) or gives back the drafter's weights; false (and `err`) when loading failed.
    virtual bool switch_drafter(bool on, std::string& err) = 0;
    /// Prints the INFO lines that start the protocol: `slots` batch slots, `spec` rows a conversation's step may run
    /// (0: no drafting).
    virtual void print_info(int slots, int spec) const = 0;
    /// Prints the DONE line that ends a request, in the fields the model's old engine wrote (the server reads each
    /// engine's).
    virtual void print_done(const RequestDone& done) const = 0;
    /// Called after each DONE and BDONE line (a batch request has both): DeepSeek writes its expert profile back, as its
    /// old engine does after each request.
    virtual void end_request() {}
    /// Allows a family to replace individual proposals with a shared network forward pass.
    virtual bool has_batched_drafter() const { return false; }
    /// Fills the shared step's requests using each session's own conversation state.
    virtual bool propose_batch(std::vector<strata::spec::DraftRequest>&, std::string&) { return false; }
};

}  // namespace strata::serve
