// src/common/serve/paged_engine.cpp - the paged engine: conversations in page pools, several decoding at once, over
// any PagedModel (paged_engine.hpp).
#include "strata/common/serve/paged_engine.hpp"
#include "strata/common/serve/request_output.hpp"

#include "strata/base/gpu/memory_info.hpp"
#include "strata/base/gpu/on_device.hpp"
#include "strata/common/experts/card_rules.hpp"
#include "strata/common/experts/expert_swaps.hpp"
#include "strata/common/experts/ram_tier.hpp"
#include "strata/common/experts/split_rules.hpp"
#include "strata/common/serve/decode_selection.hpp"
#include "strata/common/serve/conversation_residency.hpp"
#include "strata/common/serve/loop.hpp"
#include "strata/common/serve/token_budget.hpp"
#include "strata/common/state/pause_file.hpp"
#include "strata/common/state/progress.hpp"
#include "strata/common/state/state_bytes.hpp"
#include "strata/common/state/resume_point.hpp"

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

using strata::serve::print_response;

namespace strata::serve {
namespace {

using Clock = std::chrono::steady_clock;

template <class F>
auto time_step_if(bool enabled, double& elapsed, F&& operation) -> decltype(operation());

/// Gets the milliseconds from `t0` to now.
double get_ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

/// CHANGE_SETTINGS' fields: names and values, in order.
using SettingValues = std::vector<std::pair<std::string, std::string>>;

/// A place for a conversation on the GPUs: its state on each card and the tokens in it.  Its state on each card is the
/// model's part of the same index there (PagedModel::make_parts).
struct Held {
    explicit Held(strata::serve::TokenBudget& budget) : gpu_tokens(budget) {}
    strata::serve::TokenBudget::Reservation gpu_tokens; ///< Logical tokens claimed in the shared GPU allowance.
    uint64_t image_key = 0;             ///< the content identity of the pictures in live
    int session = 0;                    ///< its index: its session in the model; its id in each card's pool - 1
    std::optional<strata::spec::DraftPolicy> policy; ///< Learned window costs of this logical conversation.
    PagedParts parts;                   ///< its state on each card, in the cards' order
    std::vector<int32_t> live;          ///< the tokens its state holds: the prompt, then the fed outputs
    strata::spec::Decoding dec;         ///< its decoding, while `decoding`
    bool decoding = false;
    std::string request_id;            ///< The current RUN identity; cached state survives its completion.
    int64_t prompt_tokens = 0, resumed = 0, prompt_read = 0;
    double prompt_ms = 0;
    int position = -1;                  ///< Scheduler position while active; never part of the wire protocol.
    Clock::time_point t0;               ///< when it started decoding
    /// the request count when it last served a request: the oldest goes first, as an idle one to pause and among
    /// decoding ones of the same priority
    uint64_t used = 0;
    uint64_t last_tokens_at = 0; ///< Unix seconds when prompt or output tokens last reached this conversation.

    /// Gets its place in the order in which conversations give way to others (smallest first): the lowest priority,
    /// then the least recently used.
    std::pair<int64_t, uint64_t> get_yield_order() const { return {dec.priority, used}; }

    /// Forgets its tokens and gives back its growing pages.
    void forget() {
        live.clear();
        gpu_tokens.shrink_to(0);
        last_tokens_at = 0;
        policy.reset();
        image_key = 0;
        for (const std::unique_ptr<PagedPart>& p : parts)
            if (p->is_ready())
                p->truncate(0);
    }
    /// Tells whether it has its pages on every card (PagedPart::init).
    bool has_pages() const {
        return std::all_of(parts.begin(), parts.end(),
                           [](const std::unique_ptr<PagedPart>& p) { return p->is_ready(); });
    }
    /// Tells whether any card still holds this conversation's fixed state, including an empty cache.
    bool is_resident() const {
        return std::any_of(parts.begin(), parts.end(), [](const auto& part) { return part->is_ready(); });
    }
    /// Returns the KV pages and live recurrence to their pools, and releases host checkpoints.
    void release() {
        forget();
        for (const auto& part : parts)
            if (part->is_ready())
                part->release();
    }
    /// Gets the bytes of the pool pages it holds on every card.
    uint64_t get_bytes() const {
        uint64_t bytes = 0;
        for (const std::unique_ptr<PagedPart>& p : parts)
            bytes += p->bytes();
        return bytes;
    }
};

/// A conversation paused out of the GPU's pool: its pages in the RAM's pool or in a pause file on the SSD, and what the
/// model kept beside them.
struct Paused {
    std::optional<strata::spec::DraftPolicy> policy; ///< Restored in-process; session files rebuild the cost history.
    uint64_t serial = 0;              ///< its number (LIST_CONVERSATIONS' parked<serial>) and its id in the RAM's pool
    /// its tokens, the positions short of its end it can go back to (get_rewind_points), the model's state beside the
    /// pages (save_state) and, on the SSD, the layer cut its file has the cards' sections at (get_cut)
    strata::core::PauseHeader head;
    int64_t priority = 0;             ///< its last request's
    uint64_t used = 0;                ///< Held::used
    uint64_t bytes = 0;               ///< its pages' bytes
    std::string file;                 ///< on the SSD: its pause file (get_pause_file); empty: in the RAM's pool
};

/// Keeps admission metadata without allocating conversation state while capacity is occupied.
struct PendingRequest {
    std::string line, id;
    int64_t priority = 0, prompt_tokens = 0;
};

/// The engine once its model is up: its conversations, the pages they share with the experts, the step that decodes
/// them, and the requests' protocol.
class PagedEngine {
public:
    /// See serve_paged_model.
    PagedEngine(PagedModel& model, PagedCards& cards, strata::core::RamExperts& ram, bool copy_back,
                const PagedSettings& settings);
    /// Drains state writes and deletes the paused conversations' files.
    ~PagedEngine();

    /// Opens the shared memory pools and runtime resources; conversations are created on admission.
    bool init();
    /// Answers the server's commands until QUIT or the end of stdin; returns the exit status.
    int serve();

private:
    /// Changes the settings CHANGE_SETTINGS names (memory_report.hpp): the pools' sizes, the token limits, the SSD's
    /// budget, the drafter, the conversations at once, the rows and the lookahead.  Gets why not, having changed
    /// nothing, or an empty string.  When a memory gives less than asked (a failed allocation), the change is made as
    /// far as it goes, the report shows the sizes it reached, and `shortfall` says what was missing.
    std::string change_settings(const SettingValues& values, std::string& shortfall);
    /// Takes or gives back chunks of `card`'s pool so that it holds `bytes`.  When the conversations' pages do not fit
    /// the smaller pool, conversations are paused until they do: those that do not decode first (those in the chunks
    /// it gives back before the others), then batch slots' decoding ones in those chunks, the lowest priority first,
    /// which end as preempted.  Gets why not (the memory had less to give), or an empty string.
    std::string resize_gpu(PagedCard& card, uint64_t bytes);
    /// Takes or gives back chunks of the RAM's pool so that it (and the staging ring) hold `bytes`: the conversations
    /// paused in the chunks it gives back move to the SSD.  Gets why not, or an empty string.
    std::string resize_ram(uint64_t bytes);
    void report_resize(const char* name, const strata::core::PagePoolReport& before,
                       const strata::core::PagePoolReport& after, Clock::time_point t0) const;
    /// Runs the layer split's planner on the cards' pool sizes (their memory settings), for the conversations that
    /// decode and the token minimum; gets the cut it chose (get_cut's form), or today's when no other can run (a card
    /// would lack a layer's dense weights, or the model cannot run its layers: PagedModel::can_run_layers).
    std::vector<int64_t> plan_cut() const;
    /// Moves the layer cut to `cut` between two steps: every conversation is paused into the RAM's pool, each card
    /// takes its new layers (the pool's layout for them and their experts, the conversations' state and the model's
    /// side of the card), the decoding conversations come back, and each card fills its pool with its experts before
    /// the next step, so that the steps compute the same experts on the GPUs as before.  False (and `err`) when a
    /// conversation did not fit in the RAM's pool, or a pool cannot have the new layout: the cut stays.
    bool move_cut(const std::vector<int64_t>& cut, std::string& err);
    /// Updates model ownership while conversations have no GPU pages.
    bool set_layers(const std::vector<int64_t>& cut, std::string& err);
    /// Gives each card's pool the experts it wants that fit (refill), waiting for pages that open soon; then uploads
    /// the residency tables.
    bool fill_cards(std::string& err);
    /// Gives each card's pool the experts it is to hold (ExpertPages::set_ranking): its layers' share of the profile's
    /// first --expert-cache (PagedModel::get_card_ranking), or with `lend`, while the cards lend pages (holder_),
    /// those that place_experts puts on it, which may be of another card's layers that their card has no room for,
    /// where a hop beats the CPU (is_hop_wanted).  The experts it no longer holds leave now; refills bring in the
    /// others.
    void place_card_experts(bool lend);
    /// Tells whether a card's pool may hold the experts of another card's layers, which `holder` then computes in one
    /// hop: always with --remote-experts on; with auto, when one hop over its link costs less than the CPU computing
    /// one expert (PagedModelFacts::cpu_expert_ms).
    bool is_hop_wanted(const PagedCard& holder) const;
    /// Gets the layer cut: the first layer of each card after the first.
    std::vector<int64_t> get_cut() const;
    /// Applies the shared GPU allowance, per-conversation ceiling and pools' token minimum, pausing GPU state that
    /// exceeds them. The RAM pause allowance also follows `max`. Gets why it could not apply, or an empty string.
    std::string set_token_limits(int64_t min, int64_t max, int64_t context);
    /// Loads (`on`) or gives back the drafter's weights, and keeps settings_.drafter to whether they are loaded; false
    /// (and `err`) when loading failed.
    bool switch_drafter(bool on, std::string& err);
    /// Gets the least `card`'s pool may hold now: the pages of a conversation token minimum of `tokens_min`, and
    /// those each decoding conversation takes there at the end of its request, in whole chunks.
    uint64_t get_smallest_gpu(const PagedCard& card, int64_t tokens_min) const;
    /// Gets the most `card`'s pool may hold now: what it holds and what its GPU has free beyond what it keeps.
    uint64_t get_largest_gpu(const PagedCard& card) const;
    /// Gets the least and the most the RAM's pool and its staging ring may hold now.
    uint64_t get_smallest_ram() const;
    uint64_t get_largest_ram() const;
    /// Validates RUN, then queues it or admits a queued request at a model boundary.
    bool serve_request(const std::string& line, bool admit = false);
    /// Admits the highest-priority waiting request when capacity permits; cancellations take precedence.
    bool admit_pending_request();
    /// Allocates state only when admission needs another conversation.
    bool create_conversation(std::string& err);
    /// Emits one terminal record and releases request transport state.
    void finish_request(Held& h, const char* finish);
    /// Runs one `SAVE <path>` or `RESTORE <path>` line (the path runs to the line's end) between requests, and answers
    /// with SAVED or RESTORED, or SERR when it refused or failed with nothing changed.  The server's /slots/0 sends
    /// them; the "session" is the conversation most recently used, on the GPUs or paused.
    void run_session_command(const std::string& line);
    /// Writes the conversation most recently used (one that does not decode) to the pause file `path`, so that a
    /// RESTORE of it, also by another run of the engine, continues that conversation.  When there is none or the file
    /// could not be written, `path` is as it was.
    void save_session(const std::string& path);
    /// Reads the pause file `path` into the paused conversations, as the most recently used: the next request whose
    /// prompt starts with its tokens continues it, reading only the rest.  When the file was refused or could not be
    /// copied, nothing changed.
    void restore_session(const std::string& path);
    /// Claims compatible orphaned SSD pauses, the most recently active ranked highest, and says why it leaves each
    /// other pause file in `dir` (unreadable, foreign, or its engine still runs); those stay untouched.
    void adopt_pause_files(const std::string& dir);
    /// Moves every conversation in the GPUs' and the RAM's pools to the SSD, the highest ranked first (find_lowest's
    /// order) up to --ssd-pause-mib, and lets go of the files so that ~PagedEngine leaves them for the next start
    /// (PagedSettings::save_on_shutdown).
    void save_conversations_for_restart();
    /// Prints a SERR line (SessionError `kind`, `why`) for a refused or failed SAVE or RESTORE `action` of `path`, and
    /// says it on stderr.
    void refuse_session(const char* action, const std::string& path, strata::core::SessionError kind,
                        std::string why) const;
    /// Announces on stdout a step of a SAVE or RESTORE that blocks without output, over `bytes`: `phase` (one word) and
    /// the seconds the server allows it (SWAIT).
    void announce_wait(const char* phase, uint64_t bytes) const;
    /// Gets the identity files are bound to into identity_ unless it is there: the model's (PagedModel::
    /// get_session_identity) and the layout of the pages.  False (and `err`) when the model files cannot be read.
    bool find_identity(std::string& err);
    /// Reads `ids[from, to)` into `h` (the request began at `t0`); `read_to` gets how far it got.  Conversations that
    /// decode meanwhile take steps between the prompt's turns (PagedSettings::decode_share_percent and
    /// prompt_turn_ms).  False with `err` "cancelled" after a STOP, else on a failure that ends the engine (`fatal`:
    /// a decode step failed, its ERR printed).
    bool read_prompt(Held& h, const std::vector<int64_t>& ids, int64_t from, int64_t to, Clock::time_point t0,
                     int64_t& read_to, bool& fatal, std::string& err);
    /// Runs one step of the selected conversations and prints what it emitted; false (ERR printed) ends the engine.
    bool decode_step();
    /// Gets the unfinished requests selected by priority and request order for the next decode step.
    std::vector<Held*> get_decoding_conversations() const;
    /// Counts the next token of each selected request beyond the prefixes already claimed.
    uint64_t get_decode_growth(const std::vector<Held*>& selected) const;
    /// Keeps a prompt read from preempting higher-ranked decoders just to interleave another step.
    bool can_decode_while_reading() const;
    /// Checks admission against immediately reclaimable idle or lower-priority GPU state.
    bool can_admit_conversation(int64_t tokens, int64_t priority) const;
    /// Gets how many conversations own fixed state on any GPU; a layer split counts once.
    int get_resident_conversations() const;
    /// Tests activity age only for GPU caches outside prompt processing and decoding.
    bool is_stale(const Held& h) const;
    /// Finds idle GPU state eligible for age-based reclamation, preserving `keep`.
    Held* find_stale(const Held* keep = nullptr) const;
    /// Reclaims old idle state before an allocation would displace experts from this card.
    bool reclaim_stale_gpu_state(PagedCard& card, uint64_t free_bytes, const Held* keep, std::string& err);
    /// Checks the live decode and cache counts against the engine's capacity.
    std::string get_conversation_count_error(const PagedSettings& settings) const;
    /// Pauses idle caches, then eligible running requests, until the GPU holds at most `maximum` conversations.
    bool trim_gpu_conversations(int maximum, const Held* keep = nullptr,
                                std::optional<int64_t> priority = std::nullopt);
    /// Reports failures while draining state before commands or completed requests expose it.
    bool wait_model_state();
    /// Puts the conversation that holds the longest start of `ids` (an idle one on the GPU, or a paused one, which
    /// comes back) in place `target` of `at_`, pausing the one there unless that is it.  A conversation whose tokens
    /// `ids` leave before their end comes back at the latest point it can go back to (get_resume_tokens), and stays
    /// paused whole as well, as strata-dsv41's slots keep it.  Gets how many of `ids` it holds.
    int64_t take_conversation(size_t target, const std::vector<int64_t>& ids, uint64_t image_key, int64_t priority);
    /// Gives `h` the pages for `tokens` tokens on every card, pausing idle conversations (the lowest priority, then
    /// the least recently used, first) while a pool has no room, and tells the kernels about the experts that left
    /// those pages.
    bool make_room(Held& h, int64_t tokens, std::string& err,
                   std::optional<int64_t> priority = std::nullopt);
    /// Finds the newest request at the lowest priority, optionally below an incoming request.
    Held* find_decoding_victim(const Held* keep = nullptr, std::optional<int64_t> priority = std::nullopt);
    /// Returns a conversation's GPU claims while keeping its state in a lower tier.
    void pause_for_capacity(Held& h);
    /// Pauses `h`'s conversation, or forgets it when that fails (said on stderr).
    void pause_or_forget(Held& h);
    /// Ends the request at a scheduler position so its cached prefix can resume later.
    void end_preempted(size_t position);
    /// Finds idle GPU state to release: empty first, then the lowest priority and oldest use, among those `can_go`
    /// accepts besides `keep` and the one whose prompt is read; null when none.
    Held* find_idle(const Held* keep, const std::function<bool(const Held&)>& can_go = nullptr) const;
    /// Gives `h` its fixed regions in the pools of the cards where it has none.
    bool init_pages(Held& h, std::string& err, std::optional<int64_t> priority = std::nullopt);
    /// Moves `h`'s conversation out of the cards' pools into the RAM's (first moving the RAM's paused conversations
    /// down to the SSD while they would take more than --tokens-max allows), or else into files, one per card; `h`
    /// holds nothing after.  `ram_only`: into the RAM's pool beyond that budget, or not at all (move_cut).  False (and
    /// `err`) when a copy failed or the RAM's pool had no room for it: then `h` keeps its conversation.
    bool pause(Held& h, std::string& err, bool ram_only = false);
    /// Puts paused conversation `p` back in `h`, which holds nothing: the pages of its first `tokens` tokens (a request
    /// that branches off before its end needs no more room than that), then the model's state beside them
    /// (PagedModel::restore_state).  Short of its end, the caller then rewinds it to `tokens`.
    bool resume(const Paused& p, Held& h, int64_t tokens, std::string& err,
                std::optional<int64_t> priority = std::nullopt);
    /// Moves the lowest-priority conversation paused in RAM to the SSD (forgetting it when the copy fails); false when
    /// none is there.
    bool move_down();
    /// Moves paused conversation `p`, in RAM, to the SSD (forgetting it when the copy fails); false when the copy
    /// failed.
    bool move_to_ssd(std::vector<Paused>::iterator p);
    /// Moves the lowest-priority conversations paused in RAM to the SSD while those in RAM take more than the budget.
    void trim_ram();
    /// Forgets the lowest-priority paused conversations while those on the SSD take more than --ssd-pause-mib.
    void trim_ssd();
    /// Gives back a paused conversation's pages in RAM, or deletes its files.
    void drop_pages(const Paused& p);
    /// Gets the bytes of the paused conversations in RAM (`in_ram`) or on the SSD.
    uint64_t get_paused_bytes(bool in_ram) const;
    /// Gets the bytes the conversations paused in RAM may take: one conversation of --tokens-max tokens.
    uint64_t get_ram_pause_budget() const;
    /// Gets the pause file of the conversation of Paused::serial `serial`.
    std::string get_pause_file(uint64_t serial) const;
    /// Gets where the pages of the conversation held in place `h` are in each card's pool, for a pause file's sections.
    std::vector<strata::core::PauseSection> get_gpu_sections(const Held& h) const;
    /// Gets where the pages of the conversation paused in the RAM's pool as `serial` are, per card at the layer cut.
    std::vector<strata::core::PauseSection> get_ram_sections(uint64_t serial) const;
    /// Writes the paused conversation at its saved layer cut, with one section per card.
    /// Returns false with `err` when fingerprinting or writing fails.
    bool save_pause_file(const Paused& p, const std::string& path,
                         const std::vector<strata::core::PauseSection>& sections, std::string& err);
    /// Gets the paused conversation that goes first, the lowest priority then the least recently used, among those in
    /// RAM (`in_ram`) or on the SSD; paused_.end() when there is none.
    std::vector<Paused>::iterator find_lowest(bool in_ram);
    /// Every --adapt-every steps (the old engine's adaptive tier, with its options): counts a use, in the GPUs' pools,
    /// of the experts there that the steps since the last round routed to; then the experts decoding routed to most
    /// that the pools do not hold take the pages of the least-routed ones of their layer (at most --adapt-swaps,
    /// each by --adapt-min-gain more routings).  False (ERR printed) when a copy failed.
    bool adapt_experts();
    /// Puts back a batch of the experts the conversations evicted (the GPUs' first, then the RAM's), or waits until
    /// pages open to them: the serve loop's work between requests.  False when a copy failed (said on stderr).
    bool refill_experts();
    /// Tells whether a pool wants experts back (ExpertPages::is_refill_wanted).
    bool is_refill_wanted() const;
    /// Uploads the cards' residency tables when experts came or left since the last upload.
    bool publish_residency(std::string& err);
    /// Carries targeted cancellations into the scheduler before its next model step.
    void poll_cancellations();
    /// Tells whether any conversation decodes.
    bool is_decoding() const;
    /// Tells whether the drafter is switched on (the model has one and settings_.drafter).
    bool is_model_drafting() const { return settings_.drafter && facts_.has_drafter; }
    /// Gets the most conversations one step of this model can run.
    int get_most_concurrent() const {
        const int slots = std::min(facts_.max_rows, 8);
        return facts_.max_concurrent > 0 ? std::min(slots, facts_.max_concurrent) : slots;
    }
    /// Gives the step settings_' step settings, the lookahead 0 while the drafter is off.
    void apply_step_settings();
    /// Gets the conversation in place `place` of `at_`.
    Held& get_held(size_t place) const { return *held_[at_[place]]; }
    /// Gets the card that runs layer `layer`.
    PagedCard& get_card_of(int64_t layer) const {
        return *std::find_if(cards_.begin(), cards_.end(), [&](const PagedCard& c) { return layer < c.le; });
    }
    /// Tells whether a card other than the one that runs `layer` holds its expert `expert` (PagedCard::lent).
    bool is_lent(int64_t layer, int64_t expert) const {
        const int32_t holder = holder_.empty() ? -1 : holder_[(size_t) layer];
        return holder >= 0 && cards_[(size_t) holder].lent[(size_t) (layer * facts_.n_expert + expert)] >= 0;
    }
    /// Describes the conversations for LIST_CONVERSATIONS and GET_CONTEXT_WINDOWS.
    std::vector<strata::core::HeldConversation> list_conversations() const;
    /// Describes the memories for GET_MEMORY_USE.
    strata::serve::MemoryReport describe_memory() const;
    /// Gets the step's view of the model: PagedModel's rows over the conversations' state.
    strata::spec::StepModel get_step_model();
    /// Finds learned costs by token affinity even when the GPU state cannot rewind to that prompt.
    std::optional<strata::spec::DraftPolicy> find_draft_policy(const std::vector<int64_t>& ids,
                                                             uint64_t image_key) const;
    /// Says `format`'s line on stderr after the program's name.
    void say(const char* format, ...) const;

    PagedModel& model_;
    const PagedModelFacts& facts_;   ///< model_.facts
    PagedCards& cards_;
    strata::core::RamExperts& ram_;
    PagedSettings settings_;
    strata::serve::Input input_;
    strata::spec::Step step_;
    strata::serve::TokenBudget gpu_token_budget_; ///< Outlives the reservations owned by held_.
    std::vector<std::unique_ptr<Held>> held_;   ///< by session index
    /// Maps scheduler positions to lazily created conversation state.
    std::vector<size_t> at_;
    std::vector<PendingRequest> pending_; ///< Requests waiting for scheduler or GPU capacity.
    std::vector<Paused> paused_;       ///< the conversations paused out of the GPU's pool
    uint64_t serial_ = 0;              ///< the last paused conversation's Paused::serial
    std::string pause_prefix_;         ///< a paused conversation's file is this, its serial and ".bin"
    /// what pause files are bound to (find_identity): computed at the first pause to the SSD or SAVE, which reads the
    /// model files
    std::optional<strata::core::SessionFileIdentity> identity_;
    uint64_t requests_ = 0;            ///< the requests served so far (Held::used)
    const Held* reading_ = nullptr;    ///< the conversation whose prompt read_prompt reads, which make_room keeps
    strata::serve::Loop* command_loop_ = nullptr; ///< The active serve loop, used at prompt-chunk boundaries.
    std::string fatal_error_; ///< Stops serving when the model cannot recover its layer ownership.
    bool residency_changed_ = false;   ///< the model's residency table changed since the devices' were uploaded
    /// Per layer, the card whose pool holds the layer's experts that its own card's does not (the model's hop
    /// holders), or -1; empty while the cards lend no pages (one card, or --remote-experts off)
    std::vector<int32_t> holder_;
    int pipeline_peak_ = 0; ///< The most stages occupied together, reported when it increases.
    int64_t steps_ = 0;                ///< the decode steps so far (adapt_experts' rounds)
    /// decode_step refills experts too, until none are wanted: a pool grew, or is filling after the start
    /// (ExpertPages::open)
    bool refilling_ = true;
    std::vector<float> seen_;          ///< the model's usage counts when adapt_experts last decayed them
    int64_t swapped_ = 0;              ///< the experts adapt_experts moved into the GPU's pool, since the last report
    double adapt_ms_ = 0;              ///< the time those took
    int64_t copied_back_ = 0;          ///< the experts the GPU's pool evicted that were copied to RAM, since then
    double copy_back_ms_ = 0;          ///< the time the copies back took (all evictions', failed copies too)
    /// the model's usage counts before the step that runs now: the experts whose count grows were routed to in the step
    std::vector<float> routed_before_;
    int step_rows_ = 0;                ///< the rows the last step ran
    /// Host elapsed times within the current decode step; GPU waits are included in their caller's time.
    double draft_ms_ = 0, verify_ms_ = 0, commit_ms_ = 0, pages_ms_ = 0;
    /// Since the last report: the decode steps that routed on the CPU's side (not every expert on the GPU), their
    /// rows, and the experts they routed to, each counted once a step (docs/DETAILS.md, rows and guesses)
    int64_t routed_steps_ = 0, routed_rows_ = 0, routed_experts_ = 0;
};

/// Gets the layers [lb, le) card `k` runs at layer cut `cut` (PagedEngine::get_cut's form) of a model of `n_layers`.
std::pair<int64_t, int64_t> get_card_layers(const std::vector<int64_t>& cut, size_t k, int64_t n_layers) {
    return {k == 0 ? 0 : cut[k - 1], k < cut.size() ? cut[k] : n_layers};
}

/// Tells whether `cut` is a layer cut (PagedEngine::get_cut's form) of a model of `n_layers`: first layers that rise,
/// each of a card after the first.
bool is_layer_cut(const std::vector<int64_t>& cut, int64_t n_layers) {
    int64_t before = 0;
    for (const int64_t first : cut) {
        if (first <= before || first >= n_layers)
            return false;
        before = first;
    }
    return true;
}

/// Gets Windows' remaining allocation capacity, without an automatic reserve.
uint64_t get_commit_room() {
    strata::core::detail::HostMemory hm;
    return strata::core::detail::host_available_memory(hm) ? hm.commit : UINT64_MAX;
}

/// Gets `bytes` rounded to the nearest whole number of `pool`'s chunks (a size given in MiB is seldom one).
uint64_t round_to_chunks(const strata::core::PagePool& pool, uint64_t bytes) {
    return (bytes + pool.chunk_bytes() / 2) / pool.chunk_bytes() * pool.chunk_bytes();
}

}  // namespace

int serve_paged_model(PagedModel& model, PagedCards& cards, strata::core::RamExperts& ram, bool copy_back,
                      const PagedSettings& settings) {
    PagedEngine engine(model, cards, ram, copy_back, settings);
    return engine.init() ? engine.serve() : 1;
}

std::string get_step_settings_problem(const strata::spec::StepSettings& s, int max_rows) {
    if (s.rows < 1 || s.rows > max_rows)
        return "--rows must be 1 to " + std::to_string(max_rows);
    if (s.concurrent > s.rows)
        return "the conversations at once (" + std::to_string(s.concurrent) + ") must not exceed the rows (" +
               std::to_string(s.rows) + ")";
    if (s.lookahead < 0)
        return "--lookahead must be 0 or more";
    return {};
}

uint64_t get_gpu_room(int dev, uint64_t keep, uint64_t mapped) {
    const strata::core::OnDevice on(dev);
    uint64_t free_b = 0;
    if (strata::core::get_share_room(dev, mapped, free_b))
        free_b = std::min(free_b, get_commit_room());
    else
        free_b = strata::core::device_free_bytes();
    return free_b > keep ? free_b - keep : 0;
}

uint64_t get_ram_room(uint64_t headroom) {
    strata::core::detail::HostMemory hm;
    if (!strata::core::detail::host_available_memory(hm))
        return 0;
    return std::min(hm.available > headroom ? hm.available - headroom : 0, get_commit_room());
}

namespace {

PagedEngine::PagedEngine(PagedModel& model, PagedCards& cards, strata::core::RamExperts& ram, bool copy_back,
                         const PagedSettings& settings)
    : model_(model), facts_(model.facts), cards_(cards), ram_(ram), settings_(settings),
      step_(get_step_model(), model.facts.end_tokens) {
    if (settings_.tokens_max < 0)
        settings_.tokens_max = facts_.context;
    if (settings_.vram_conversations < 0) {
        settings_.vram_conversations = settings_.step.concurrent;
        uint64_t smallest = UINT64_MAX;
        for (const auto& card : cards_) {
            std::string name;
            uint64_t total = 0, free = 0;
            bool unified = false;
            strata::core::read_gpu_memory(card.dev, name, total, free, unified);
            smallest = std::min(smallest, total);
        }
        settings_.vram_conversations = strata::serve::get_default_resident_conversations(
            settings_.step.concurrent, get_most_concurrent(), smallest == UINT64_MAX ? 0 : smallest,
            facts_.extra_resident_vram);
    }
    apply_step_settings();
    if (settings_.conversation_context < 0)
        settings_.conversation_context = std::min(facts_.context, settings_.tokens_max);
    (void) gpu_token_budget_.try_set_limit((uint64_t) settings_.tokens_max);
    std::error_code ec;   // a folder that cannot be found or made fails each pause's save, which says so
    const std::filesystem::path dir = settings.pause_dir.empty()
                                          ? std::filesystem::temp_directory_path(ec) / facts_.name
                                          : std::filesystem::path(settings.pause_dir);
    std::filesystem::create_directories(dir, ec);
    pause_prefix_ = strata::core::get_pause_file_prefix(dir.string());
    try {
        adopt_pause_files(dir.string());
    } catch (const std::exception& e) {
        say("pause adoption skipped: %s\n", e.what());
    }
    for (PagedCard& card : cards_) {
        strata::core::ExpertPages::Hooks hooks;
        hooks.load = [this, &c = card](strata::core::ExpertKey key, int32_t slot, std::string& err) {
            return model_.load_expert(c, key, slot, err);
        };
        // an expert of another card's layers goes to the card's `lent`, not to the residency table its layer's card
        // reads; an evicted one leaves whichever holds it
        hooks.moved = [this, &c = card](strata::core::ExpertKey key, int32_t slot) {
            const size_t i = (size_t) key.layer * (size_t) facts_.n_expert + (size_t) key.expert;
            const bool own = c.lent.empty() || (slot >= 0 ? key.layer >= c.lb && key.layer < c.le : c.lent[i] < 0);
            (own ? model_.get_residency()[i] : c.lent[i]) = slot;
            residency_changed_ = residency_changed_ || own;
        };
        if (copy_back)
            hooks.leaving = [this](strata::core::ExpertKey key, const uint8_t* bytes) {
                const Clock::time_point t0 = Clock::now();
                copied_back_ += ram_.keep_copy(key, bytes);
                copy_back_ms_ += get_ms_since(t0);
            };
        card.experts.set_hooks(std::move(hooks));
    }
}

PagedEngine::~PagedEngine() {
    (void) wait_model_state();
    for (const Paused& p : paused_)
        drop_pages(p);
}

bool PagedEngine::init() {
    if (const std::string why = get_conversation_count_error(settings_); !why.empty()) {
        say("%s\n", why.c_str());
        return false;
    }
    if (const std::string why = strata::serve::get_token_limits_error(settings_.conversation_context, facts_.context,
            settings_.tokens_min, settings_.tokens_max); !why.empty()) {
        say("%s\n", why.c_str());
        return false;
    }
    if (cards_.size() < 2 || settings_.remote_experts == "off")
        return true;
    std::vector<int> devices;
    for (const PagedCard& card : cards_)
        devices.push_back(card.dev);
    if (const std::string why = strata::core::get_hop_refusal(devices); !why.empty()) {
        if (settings_.remote_experts == "on") {
            say("--remote-experts on: %s; use --remote-experts auto or off\n", why.c_str());
            return false;
        }
        return true;
    }
    if (facts_.split_ranking.empty())
        return true;
    for (PagedCard& card : cards_) {
        card.lent.assign((size_t) (facts_.n_layers * facts_.n_expert), strata::core::kNotResident);
        if (std::string err; !model_.attach_hop(card, err)) {
            say("%s cannot compute other cards' experts: %s\n", card.id.c_str(), err.c_str());
            for (PagedCard& c : cards_)
                c.lent.clear();
            return settings_.remote_experts != "on";   // (auto: the cards lend no pages)
        }
    }
    holder_.assign((size_t) facts_.n_layers, -1);
    model_.set_hop_holders(holder_.data());
    place_card_experts(true);
    return true;
}

int PagedEngine::serve() {
    const Clock::time_point fill_start = Clock::now();
    std::string err;
    if (!fill_cards(err)) {
        say("filling the GPU expert pools before READY: %s\n", err.c_str());
        return 1;
    }
    refilling_ = is_refill_wanted();
    say("GPU expert pools filled before READY in %.0f ms\n", get_ms_since(fill_start));
    model_.print_info(get_most_concurrent(), step_.settings.lookahead > 0 ?
                         std::min(step_.settings.rows, step_.settings.lookahead + 1) : 0);
    print_response("INFO concurrent=%d\n", settings_.step.concurrent);
    print_response("INFO vram_conversations=%lld\n", (long long) settings_.vram_conversations);
    print_response("INFO conversation_idle_seconds=%lld\n", (long long) settings_.conversation_idle_seconds);
    print_response("INFO save_on_shutdown=%d\n", settings_.save_on_shutdown ? 1 : 0);
    print_response("INFO tokens_max=%lld\n", (long long) settings_.tokens_max);
    print_response("INFO context_capacity=%lld\n", (long long) facts_.context);
    if (settings_.batch_groups != "1") {
        print_response("INFO batch_groups=%d\n", step_.settings.groups);
        if (step_.settings.groups == 1)
            say("--batch-groups %s: one group (needs stage launch/poll support and a divisor of live concurrency)\n",
                settings_.batch_groups.c_str());
    }
    print_response("INFO request_capacity=%d request_queue=1\n", get_most_concurrent());
    print_response(
        "READY %lld stop conversations memory preempt settings requests\n", (long long) settings_.conversation_context);
    strata::serve::EngineHooks hooks;
    hooks.before_line = [this](const std::string&) { return wait_model_state(); };
    hooks.is_busy = [this] { return !pending_.empty() || is_decoding() || is_refill_wanted(); };
    hooks.step = [this] {
        return fatal_error_.empty() && admit_pending_request() &&
               (is_decoding() ? decode_step() : !is_refill_wanted() || refill_experts());
    };
    hooks.list_conversations = [this] { return list_conversations(); };
    hooks.describe_memory = [this] { return describe_memory(); };
    hooks.change_settings = [this](const SettingValues& values, std::string& shortfall) {
        return change_settings(values, shortfall);
    };
    strata::serve::Loop loop(input_, hooks);
    command_loop_ = &loop;
    std::string line;
    while (loop.next(line)) {
        if (line == "QUIT") break;
        if (line.rfind("RUN ", 0) == 0 || line.rfind("RUNI ", 0) == 0) {
            if (!serve_request(line)) return 1;
            continue;
        }
        if (line.rfind("SAVE ", 0) == 0 || line.rfind("RESTORE ", 0) == 0) {
            run_session_command(line);
            continue;
        }
        print_response("ERR %s does not take %s\n", facts_.name.c_str(), line.substr(0, line.find(' ')).c_str());
    }
    command_loop_ = nullptr;
    if (loop.is_failed())
        return 1;
    if (settings_.save_on_shutdown)
        save_conversations_for_restart();
    return 0;
}

void PagedEngine::run_session_command(const std::string& line) {
    const bool save = line[0] == 'S';
    const std::string path = line.substr(save ? 5 : 8);
    try {
        if (path.empty())
            refuse_session(save ? "save" : "restore", path, strata::core::SessionError::invalid, "missing path");
        else if (save)
            save_session(path);
        else
            restore_session(path);
    } catch (const std::bad_alloc&) {
        refuse_session(save ? "save" : "restore", path, strata::core::SessionError::memory, "not enough host memory");
    } catch (const std::filesystem::filesystem_error& e) {
        refuse_session(save ? "save" : "restore", path, strata::core::SessionError::io, e.what());
    }
}

void PagedEngine::save_session(const std::string& path) {
    using strata::core::SessionError;
    const Clock::time_point t0 = Clock::now();
    const auto refuse = [&](SessionError kind, const std::string& why) { refuse_session("save", path, kind, why); };
    const Held* held = nullptr;
    for (const std::unique_ptr<Held>& x : held_)
        if (!x->decoding && !x->live.empty() && (held == nullptr || x->used > held->used))
            held = x.get();
    const auto latest = std::max_element(paused_.begin(), paused_.end(),
                                         [](const Paused& a, const Paused& b) { return a.used < b.used; });
    const Paused* paused =
        latest != paused_.end() && (held == nullptr || latest->used > held->used) ? &*latest : nullptr;
    if (held == nullptr && paused == nullptr)
        return refuse(SessionError::invalid, "no complete session to save");
    const size_t tokens = (paused != nullptr ? paused->head.live : held->live).size();
    const uint64_t estimated_bytes = paused != nullptr ? paused->bytes :
        held->get_bytes() + model_.get_host_state_bytes(held->parts);
    std::error_code ec;
    const auto parent = std::filesystem::absolute(std::filesystem::u8path(path)).parent_path();
    const std::filesystem::space_info room = std::filesystem::space(parent, ec);
    if (!ec && room.available < estimated_bytes)
        return refuse(SessionError::storage,
                       "not enough room on the drive for " + std::to_string(estimated_bytes >> 20) + " MiB");
    std::string err;
    if (!identity_)
        announce_wait("fingerprint", 2u << 20);
    if (!find_identity(err))
        return refuse(SessionError::io, err);
    announce_wait("write", estimated_bytes);
    Paused saved;
    bool ok = true;
    if (paused != nullptr && !paused->file.empty()) {
        ok = strata::core::copy_pause_file(paused->file, path, err);
    } else {
        if (paused != nullptr) {
            saved.head = paused->head;
        } else {
            saved.head.live = held->live;
            saved.head.last_tokens_at = held->last_tokens_at;
            saved.head.image_key = held->image_key;
            saved.head.points = model_.get_rewind_points(held->parts);
            ok = model_.save_state(held->parts, saved.head.state, err);
        }
        saved.head.cut = get_cut();
        ok = ok && save_pause_file(
                   saved, path, paused != nullptr ? get_ram_sections(paused->serial) : get_gpu_sections(*held), err);
    }
    if (!ok)
        return refuse(SessionError::io, err);
    const uint64_t bytes = std::filesystem::file_size(std::filesystem::u8path(path), ec);
    say("session saved: %zu tokens, %llu bytes to %s in %.1f ms\n", tokens, (unsigned long long) bytes, path.c_str(),
        get_ms_since(t0));
    print_response("SAVED %zu %llu %.1f\n", tokens, (unsigned long long) bytes, get_ms_since(t0));
    std::fflush(stdout);
}

void PagedEngine::restore_session(const std::string& path) {
    using strata::core::SessionError;
    const Clock::time_point t0 = Clock::now();
    const auto refuse = [&](SessionError kind, const std::string& why) { refuse_session("restore", path, kind, why); };
    std::string err;
    if (!identity_)
        announce_wait("fingerprint", 2u << 20);
    if (!find_identity(err))
        return refuse(SessionError::io, err);
    Paused p;
    std::error_code ec;
    const uint64_t file_bytes = std::filesystem::file_size(std::filesystem::u8path(path), ec);
    announce_wait("read", ec ? 0 : file_bytes);
    strata::core::PauseStream file;
    if (const SessionError kind = strata::core::open_pause_file(
            path, *identity_, (uint64_t) facts_.context, file, p.head, err);
        kind != SessionError::none)
        return refuse(kind, err);
    file.reset();
    if (!is_layer_cut(p.head.cut, facts_.n_layers))
        return refuse(SessionError::invalid, path + ": damaged");
    p.bytes = file_bytes;
    uint64_t protected_bytes = 0;
    for (const Paused& x : paused_)
        if (!x.file.empty() && x.priority > p.priority)
            protected_bytes += x.bytes;
    if (p.bytes + protected_bytes > ((uint64_t) settings_.ssd_pause_mib << 20))
        return refuse(SessionError::storage,
                      "not enough room under --ssd-pause-mib after higher-priority conversations");
    announce_wait("copy", p.bytes);
    p.serial = ++serial_;
    p.file = get_pause_file(p.serial);
    if (!strata::core::copy_pause_file(path, p.file, err)) {
        drop_pages(p);
        return refuse(SessionError::io, err);
    }
    std::vector<uint8_t>().swap(p.head.state);
    p.used = ++requests_;   // as if its conversation had just served a request
    const size_t tokens = p.head.live.size();
    const uint64_t serial = p.serial, bytes = p.bytes;
    paused_.push_back(std::move(p));
    trim_ssd();
    if (std::none_of(paused_.begin(), paused_.end(), [&](const Paused& x) { return x.serial == serial; }))
        return refuse(SessionError::storage, "the paused conversations of higher priority take the room");
    say("session restored: %zu tokens, %llu bytes from %s in %.1f ms\n", tokens, (unsigned long long) bytes, path.c_str(),
        get_ms_since(t0));
    print_response("RESTORED %zu %llu %.1f\n", tokens, (unsigned long long) bytes, get_ms_since(t0));
    std::fflush(stdout);
}

void PagedEngine::adopt_pause_files(const std::string& dir) {
    std::vector<std::string> running;
    const std::vector<std::string> stale = strata::core::find_stale_pause_files(dir, &running);
    for (const std::string& path : running)
        say("pause file %s not adopted: the engine that wrote it still runs\n", path.c_str());
    const size_t first = paused_.size();
    for (const std::string& path : stale) {
        const auto skip = [&](const std::string& why) {
            say("pause file %s not adopted: %s\n", path.c_str(), why.c_str());
        };
        try {
            std::string err;
            if (!find_identity(err)) {
                say("pause adoption skipped: %s\n", err.c_str());
                break;
            }
            Paused p;
            strata::core::PauseStream file;
            const strata::core::SessionError kind =
                strata::core::open_pause_file(path, *identity_, (uint64_t) facts_.context, file, p.head, err);
            if (kind != strata::core::SessionError::none) {
                skip(std::string(strata::core::session_error_name(kind)) + ": " + err);
                continue;
            }
            if (!is_layer_cut(p.head.cut, facts_.n_layers)) {
                skip("its layer cut does not fit this model's " + std::to_string(facts_.n_layers) + " layers");
                continue;
            }
            file.reset();
            std::error_code ec;
            p.bytes = std::filesystem::file_size(path, ec);
            p.serial = ++serial_;
            p.file = get_pause_file(p.serial);
            // A rename claims the file without duplicating its pages on disk.
            if (!ec)
                std::filesystem::rename(path, p.file, ec);
            if (ec) {
                skip(ec.message());
                continue;
            }
            std::vector<uint8_t>().swap(p.head.state);
            paused_.push_back(std::move(p));
        } catch (const std::exception& e) {
            skip(e.what());
        }
    }
    // the most recently active rank highest, so that trim_ssd forgets the oldest first
    std::sort(paused_.begin() + (std::ptrdiff_t) first, paused_.end(),
              [](const Paused& a, const Paused& b) { return a.head.last_tokens_at < b.head.last_tokens_at; });
    uint64_t bytes = 0;
    for (auto p = paused_.begin() + (std::ptrdiff_t) first; p != paused_.end(); ++p) {
        p->used = ++requests_;
        bytes += p->bytes;
    }
    if (paused_.size() > first)
        say("adopted %zu paused conversations (%.1f GiB) from %s\n", paused_.size() - first,
            (double) bytes / 1073741824.0, dir.c_str());
    trim_ssd();
}

void PagedEngine::save_conversations_for_restart() {
    const Clock::time_point t0 = Clock::now();
    if (!wait_model_state())
        return;
    for (const std::unique_ptr<Held>& h : held_)
        if (h->is_resident() && !h->live.empty())
            pause_for_capacity(*h);
    std::vector<std::pair<std::pair<int64_t, uint64_t>, uint64_t>> in_ram;   // (rank, serial), the highest first
    for (const Paused& p : paused_)
        if (p.file.empty())
            in_ram.push_back({{p.priority, p.used}, p.serial});
    std::sort(in_ram.rbegin(), in_ram.rend());
    const auto find = [this](uint64_t serial) {
        return std::find_if(paused_.begin(), paused_.end(), [&](const Paused& p) { return p.serial == serial; });
    };
    for (const auto& [rank, serial] : in_ram)
        // gone after a good copy: trim_ssd forgot it, so it would forget every one ranked lower too
        if (move_to_ssd(find(serial)) && find(serial) == paused_.end())
            break;
    const size_t kept =
        (size_t) std::count_if(paused_.begin(), paused_.end(), [](const Paused& p) { return !p.file.empty(); });
    const uint64_t bytes = get_paused_bytes(false);
    std::erase_if(paused_, [](const Paused& p) { return !p.file.empty(); });
    say("%zu conversations (%.1f GiB) kept on the SSD for the next start, in %.0f ms\n", kept,
        (double) bytes / 1073741824.0, get_ms_since(t0));
}

void PagedEngine::refuse_session(const char* action, const std::string& path, strata::core::SessionError kind,
                                 std::string why) const {
    say("session %s %s: %s\n", action, path.c_str(), why.c_str());
    std::replace_if(why.begin(), why.end(), [](char c) { return c == '\n' || c == '\r'; }, ' ');
    print_response("SERR %s 0 %s\n", strata::core::session_error_name(kind), why.c_str());
    std::fflush(stdout);
}

void PagedEngine::announce_wait(const char* phase, uint64_t bytes) const {
    print_response("SWAIT %s %lld\n", phase, (long long) strata::core::session_phase_limit_s(bytes));
    std::fflush(stdout);
}

bool PagedEngine::find_identity(std::string& err) {
    if (identity_)
        return true;
    strata::core::SessionFileIdentity id;
    if (!model_.get_session_identity(id, err))
        return false;
    // pages of another layout (the drafter's K/V present or not, another page size) are another configuration
    const strata::core::PagePool& ram = ram_.pages().pool();
    strata::core::SessionIdentityBuilder layout(id.config);
    layout.u64("page_bytes", ram.page_bytes());
    for (size_t r = 0; r < ram.layout().regions.size(); ++r) {
        const strata::core::StateRegion& region = ram.layout().regions[r];
        layout.u64("kind", (uint64_t) region.kind);
        layout.u64("unit_bytes", region.unit_bytes);
        layout.u64("rows", region.rows);
        layout.u64("block_bytes", ram.block_bytes(r));
    }
    id.config = layout.digest();
    identity_ = id;
    return true;
}

bool PagedEngine::serve_request(const std::string& line, bool admit) {
    strata::serve::RequestLine req;
    strata::serve::Sampling smp = facts_.sampling;
    std::vector<int64_t> ids;
    std::string err;
    if (!strata::serve::parse_request_line(line, req) || req.request_id.empty()) {
        print_response("ERR malformed RUN request\n");
        return true;
    }
    const strata::serve::RequestOutput output(req.request_id);
    if (std::any_of(held_.begin(), held_.end(), [&](const auto& h) {
            return h->decoding && h->request_id == req.request_id;
        }) || std::any_of(pending_.begin(), pending_.end(), [&](const PendingRequest& request) {
            return request.id == req.request_id;
        })) {
        input_.finish_request(req.request_id);
        print_response("ERR duplicate request identity\n");
        return true;
    }
    struct Admission {
        Input& input;
        const std::string& id;
        bool retained = false;
        ~Admission() {
            if (!retained)
                input.finish_request(id);
        }
    } admission{input_, req.request_id};
    if (req.max_new < 1 || !parse_i64_list(req.ids.c_str(), ids, err)) {
        print_response("ERR malformed RUN request\n");
        return true;
    }
    int64_t priority = 0;
    for (const auto& [key, value] : req.keys)
        if (key == "priority")
            priority = std::atoll(value.c_str());
        else
            (void) strata::serve::read_sampling_key(key, value, smp);   // the other engines' keys are not this one's
    const int64_t n = (int64_t) ids.size();
    if (input_.is_cancel_requested(req.request_id)) {
        print_response("DONE 0 %lld 0 0 cancel\n", (long long) n);
        return true;
    }
    if (n >= settings_.conversation_context) {
        print_response("ERR the prompt (%lld tokens) does not fit the per-conversation limit (%lld)\n",
                                      (long long) n, (long long) settings_.conversation_context);
        return true;
    }
    if (n >= settings_.tokens_max) {
        print_response("ERR the prompt (%lld tokens) does not fit the shared GPU token budget (%lld)\n",
                    (long long) n, (long long) settings_.tokens_max);
        return true;
    }
    if (!admit || !can_admit_conversation(n, priority)) {
        pending_.push_back({line, req.request_id, priority, n});
        admission.retained = true;
        return true;
    }
    size_t target = 0;
    while (target < at_.size() && get_held(target).decoding)
        ++target;
    if (target == at_.size() && at_.size() >= (size_t) get_most_concurrent()) {
        Held* victim = find_decoding_victim(nullptr, priority);
        if (!victim) {
            pending_.push_back({line, req.request_id, priority, n});
            admission.retained = true;
            return true;
        }
        target = (size_t) victim->position;
        end_preempted(target);
    }
    if (target == at_.size() && !create_conversation(err)) {
        print_response("ERR %s\n", err.c_str());
        return false;
    }
    print_response("STARTED\n");
    uint64_t image_key = 0;
    if (!model_.prepare_images(req, ids, image_key, err)) {
        print_response("ERR %s\n", err.c_str());
        return true;
    }
    input_.clear_stop();   // a STOP between requests is stale
    const Clock::time_point t0 = Clock::now();
    const auto policy = find_draft_policy(ids, image_key);
    const int64_t resume = take_conversation(target, ids, image_key, priority);
    Held& h = get_held(target);
    h.request_id = req.request_id;
    const int session = h.session;
    h.used = ++requests_;
    h.dec = strata::spec::Decoding{};
    h.dec.priority = priority;
    h.dec.draft_limits = facts_.draft_limits;
    for (const auto& [key, value] : req.keys) {
        const float floor = std::strtof(value.c_str(), nullptr);
        if (key == "spec")
            h.dec.draft_enabled = std::atoi(value.c_str()) != 0;
        else if (key == "dspark_conf" || key == "spec_min_p")
            h.dec.draft_limits.confidence = floor >= 0 ? floor : facts_.draft_limits.confidence;
        else if (key == "dspark_cum" || key == "spec_cum")
            h.dec.draft_limits.cumulative = floor >= 0 ? floor : facts_.draft_limits.cumulative;
    }
    if (facts_.has_drafter)
        model_.forget_drafts(session);
    if (!h.has_pages() && !init_pages(h, err, priority)) {
        print_response("ERR %s\n", err.c_str());
        return true;
    }
    if (resume == 0) {
        h.forget();
        for (const std::unique_ptr<PagedPart>& part : h.parts)
            part->clear_state();
    }
    h.policy = policy;
    print_response("RESUME %lld\n", (long long) resume);
    h.image_key = image_key;
    model_.attach_images(h.parts);
    model_.start_request(session, {smp, req.keys});
    const char* finish = "length";
    int64_t read_to = resume;   // the prompt tokens the session holds
    if (!make_room(h, n - 1, err, priority)) {   // refused: the engine stays up
        print_response("ERR %s\n", err.c_str());
        return true;
    }
    if (n - 1 > resume) {
        bool fatal = false;
        const bool ok = read_prompt(h, ids, resume, n - 1, t0, read_to, fatal, err);
        if (fatal)
            return false;
        if (!ok && err != "cancelled" && err != "settings pending") {
            print_response("ERR %s\n", err.c_str());
            return false;
        }
        if (!ok)
            finish = err == "settings pending" ? "preempted" : "cancel";
    }
    h.live.assign(ids.begin(), ids.begin() + read_to);
    if (h.gpu_tokens.get_used() > (uint64_t) read_to)
        for (const auto& part : h.parts)
            part->truncate(read_to);
    h.gpu_tokens.shrink_to((uint64_t) read_to);
    const double prompt_ms = get_ms_since(t0);
    print_response("REUSED %lld\n", (long long) resume);
    const Clock::time_point d0 = Clock::now();
    h.t0 = d0;
    if (read_to == n - 1 && std::string_view(finish) == "length") {
        h.dec.session = session;
        h.dec.next = (int32_t) ids[(size_t) (n - 1)];
        h.dec.pos = n - 1;
        h.dec.max_new = std::min<int64_t>(req.max_new, settings_.conversation_context - n);
        // Mixed sampled windows need per-row proposal distributions before lookup can be enabled.
        const bool lookup = smp.temperature <= 0.0f;
        if (h.dec.draft_enabled && (facts_.has_drafter ||
            (lookup && (settings_.suffix_draft > 0 || settings_.lookup_chain > 0))) &&
            settings_.step.lookahead > 0) {
            h.dec.planner = std::make_unique<strata::spec::WindowPlanner>(
                facts_.max_rows, settings_.step.model_lookahead < 0 ? facts_.max_rows :
                    settings_.step.model_lookahead + 1, lookup ? (int) settings_.suffix_draft : 0,
                lookup ? (int) settings_.lookup_chain : 0, (int) settings_.lookup_chain_min, facts_.context);
            if (h.policy)
                h.dec.planner->policy = *h.policy;
            h.dec.planner->start(ids);
        }

        h.decoding = true;
        h.position = (int) target;
    }
    h.prompt_tokens = n;
    h.resumed = resume;
    h.prompt_read = read_to - resume;
    h.prompt_ms = prompt_ms;
    if (!h.decoding)
        finish_request(h, finish);
    admission.retained = h.decoding;
    return true;
}

bool PagedEngine::admit_pending_request() {
    if (pending_.empty())
        return true;
    auto request = std::find_if(pending_.begin(), pending_.end(), [&](const PendingRequest& pending) {
        return input_.is_cancel_requested(pending.id);
    });
    if (request == pending_.end())
        request = std::max_element(pending_.begin(), pending_.end(), [](const auto& a, const auto& b) {
            return a.priority < b.priority;
        });
    const bool invalid = request->prompt_tokens >= settings_.conversation_context ||
                         request->prompt_tokens >= settings_.tokens_max;
    if (!input_.is_cancel_requested(request->id) && !invalid &&
        !can_admit_conversation(request->prompt_tokens, request->priority))
        return true;
    if (!wait_model_state())
        return false;
    std::string line = std::move(request->line);
    pending_.erase(request);
    return serve_request(line, true);
}

bool PagedEngine::create_conversation(std::string& err) {
    auto h = std::make_unique<Held>(gpu_token_budget_);
    h->session = (int) held_.size();
    for (PagedCard& card : cards_) {
        PagedParts parts = model_.make_parts(card, held_.size(), 1, err);
        if (parts.empty())
            return false;
        h->parts.push_back(std::move(parts.front()));
    }
    at_.push_back(held_.size());
    held_.push_back(std::move(h));
    return true;
}

void PagedEngine::finish_request(Held& h, const char* finish) {
    const strata::serve::RequestOutput output(h.request_id);
    input_.finish_request(h.request_id);
    model_.print_done({h.dec.produced, h.prompt_tokens, h.resumed, h.prompt_read, h.prompt_ms,
                       get_ms_since(h.t0), finish, h.dec.guessed, h.dec.guessed_right});
    h.decoding = false;
    model_.end_request();
}

bool PagedEngine::read_prompt(Held& h, const std::vector<int64_t>& ids, int64_t from, int64_t to,
                              Clock::time_point t0, int64_t& read_to, bool& fatal, std::string& err) {
    reading_ = &h;
    h.live.assign(ids.begin(), ids.begin() + from);
    const PagedModel::ChunkDone on_chunk = [&](int64_t pos0, int64_t T, std::string& e) -> bool {
        read_to = pos0 + T;
        h.live.insert(h.live.end(), ids.begin() + h.live.size(), ids.begin() + read_to);
        h.last_tokens_at = (uint64_t) std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        const double ms = get_ms_since(t0);
        print_response("PP %lld %lld %.0f %.1f\n", (long long) read_to, (long long) ids.size(), ms,
                    ms > 0 ? 1000.0 * (double) (read_to - from) / ms : 0.0);
        strata::core::progress_beat();
        if (command_loop_)
            command_loop_->poll_inspections();
        if (input_.is_stop_requested())
            e = "cancelled";
        else if (input_.has_pending_settings())
            e = "settings pending";
        else
            return true;
        return false;
    };
    // beside conversations that decode, a chunk at a time, and they decode between two turns of whole chunks
    const int64_t piece = std::max<int64_t>(model_.get_prompt_chunk(), 1);
    bool ok = true;
    Clock::time_point turn = Clock::now();
    for (int64_t q = from; ok && q < to; q += piece) {
        ok = model_.wait_state(err) &&
             model_.read_prompt(h.parts, ids, q, std::min(piece, to - q), is_model_drafting(), on_chunk, err);
        const double turn_ms = get_ms_since(turn);
        if (q + piece >= to || turn_ms < (double) settings_.prompt_turn_ms)
            continue;
        const double budget_ms = (double) settings_.decode_share_percent / 100.0 * turn_ms;
        const Clock::time_point td = Clock::now();
        while (ok && is_decoding() && can_decode_while_reading() && get_ms_since(td) < budget_ms) {
            poll_cancellations();
            fatal = !decode_step();
            ok = !fatal;
        }
        turn = Clock::now();
    }
    reading_ = nullptr;
    return ok;
}

bool PagedEngine::decode_step() {
    const Clock::time_point started = settings_.step_profile ? Clock::now() : Clock::time_point{};
    draft_ms_ = verify_ms_ = commit_ms_ = pages_ms_ = 0;
    poll_cancellations();
    auto going = get_decoding_conversations();
    while (get_decode_growth(going) > gpu_token_budget_.get_available()) {
        Held* victim = find_idle(nullptr);
        if (!victim)
            victim = find_decoding_victim();
        if (!victim) {
            print_response("ERR the GPU conversation budget has no room for a decode step\n");
            return false;
        }
        pause_for_capacity(*victim);
        going = get_decoding_conversations();
    }
    for (Held* h : going)
        if (!h->dec.stop)
            (void) h->gpu_tokens.try_reserve((uint64_t) (h->dec.pos + 1));
    ++steps_;
    std::vector<strata::spec::Decoding*> active;
    for (Held* h : going)
        active.push_back(&h->dec);
    const std::vector<float>& usage = model_.get_usage();
    routed_before_.assign(usage.begin(), usage.end());
    std::string err;
    if (!step_.run(active, err)) {
        print_response("ERR %s\n", err.c_str());
        return false;
    }
    if (const int64_t routed = std::inner_product(usage.begin(), usage.end(), routed_before_.begin(), (int64_t) 0,
                                                  std::plus<>(), std::greater<>());
        routed > 0) {
        routed_experts_ += routed;
        routed_rows_ += step_rows_;
        ++routed_steps_;
    }
    for (Held* h : going) {
        const strata::serve::RequestOutput output(h->request_id);
        strata::spec::Decoding& d = h->dec;
        if (d.planner)
            h->policy = d.planner->policy;
        h->live.insert(h->live.end(), d.fed.begin(), d.fed.end());
        h->gpu_tokens.shrink_to(h->live.size());
        if (!d.fed.empty() || !d.emitted.empty())
            h->last_tokens_at = (uint64_t) std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        for (const int32_t t : d.emitted)
            print_response("T %d\n", (int) t);
        if (d.finish == nullptr)
            continue;
        if (!wait_model_state())
            return false;
        h->decoding = false;
        if (d.planner)
            say("suffix drafts: %lld of %lld kept/offered\n", (long long) d.suffix_kept,
                (long long) d.suffix_offered);
        if (d.guessed > 0)
            say("%lld of %lld guesses right (%.0f%%) over %lld tokens\n", (long long) d.guessed_right,
                (long long) d.guessed, 100.0 * (double) d.guessed_right / (double) d.guessed, (long long) d.produced);
        if (routed_steps_ > 0) {
            say("%lld decode steps of %lld rows in all routed to %lld experts (each counted once a step)\n",
                (long long) routed_steps_, (long long) routed_rows_, (long long) routed_experts_);
            routed_steps_ = routed_rows_ = routed_experts_ = 0;
        }
        if (swapped_ > 0 || copy_back_ms_ > 0) {
            say("%lld experts moved into the GPU's pool (%.0f ms in all); %lld it evicted copied back to RAM (%.1f "
                "ms, all evictions)\n", (long long) swapped_, adapt_ms_, (long long) copied_back_, copy_back_ms_);
            swapped_ = copied_back_ = 0;
            adapt_ms_ = copy_back_ms_ = 0;
        }
        model_.report_hops();
        if (std::strcmp(d.finish, "nonfinite") == 0) {   // #879: its state may carry the poison: start over next time
            say("non-finite logits (#879) at position %lld: the conversation's next request reads its prompt from "
                "token 0\n", (long long) d.pos);
            h->forget();
        }
        finish_request(*h, d.finish);
    }
    strata::core::progress_beat();
    refilling_ = is_refill_wanted();
    const bool ok = time_step_if(settings_.step_profile != 0, pages_ms_,
        [&] { return adapt_experts() && (!refilling_ || refill_experts()); });
    if (settings_.step_profile)
        say("step-profile rows=%d draft=%.3f verify=%.3f commit=%.3f pages=%.3f total=%.3f ms\n",
            step_rows_, draft_ms_, verify_ms_, commit_ms_, pages_ms_, get_ms_since(started));
    return ok;
}

std::vector<Held*> PagedEngine::get_decoding_conversations() const {
    std::vector<Held*> going;
    for (const std::unique_ptr<Held>& h : held_)
        if (h->decoding)
            going.push_back(h.get());
    strata::serve::select_decode_conversations(going, facts_.serial_conversations ? 1 : settings_.step.concurrent);
    return going;
}

uint64_t PagedEngine::get_decode_growth(const std::vector<Held*>& selected) const {
    uint64_t tokens = 0;
    for (const Held* h : selected)
        if (!h->dec.stop)
            tokens += (uint64_t) std::max<int64_t>(0, h->dec.pos + 1 - (int64_t) h->gpu_tokens.get_used());
    return tokens;
}

bool PagedEngine::can_decode_while_reading() const {
    uint64_t available = gpu_token_budget_.get_available();
    for (const auto& h : held_)
        if (h.get() != reading_ && !h->decoding)
            available += h->gpu_tokens.get_used();
    return get_decode_growth(get_decoding_conversations()) <= available;
}

bool PagedEngine::can_admit_conversation(int64_t tokens, int64_t priority) const {
    uint64_t available = gpu_token_budget_.get_available();
    int active = 0;
    for (const auto& h : held_)
        if (!h->decoding || (h->position >= 0 && h->dec.priority < priority))
            available += h->gpu_tokens.get_used();
        else
            ++active;
    return active < settings_.vram_conversations && (uint64_t) tokens <= available;
}

int PagedEngine::get_resident_conversations() const {
    return (int) std::count_if(held_.begin(), held_.end(), [](const auto& h) { return h->is_resident(); });
}

bool PagedEngine::is_stale(const Held& h) const {
    const auto now = (uint64_t) std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    return h.is_resident() && strata::serve::is_conversation_stale(
        h.decoding, &h == reading_, h.last_tokens_at, now, (uint64_t) settings_.conversation_idle_seconds);
}

Held* PagedEngine::find_stale(const Held* keep) const {
    return find_idle(keep, [this](const Held& h) { return is_stale(h); });
}

bool PagedEngine::reclaim_stale_gpu_state(PagedCard& card, uint64_t free_bytes, const Held* keep, std::string& err) {
    while (card.experts.pool().report().free.bytes < free_bytes) {
        Held* stale = find_stale(keep);
        if (!stale)
            break;
        if (!model_.wait_state(err))
            return false;
        pause_or_forget(*stale);
    }
    return true;
}

std::string PagedEngine::get_conversation_count_error(const PagedSettings& settings) const {
    if (settings.conversation_idle_seconds < 0)
        return "conversation idle seconds cannot be negative";
    if (settings.vram_conversations < settings.step.concurrent)
        return "conversations held in VRAM (" + std::to_string(settings.vram_conversations) +
               ") cannot be fewer than conversations at once (" + std::to_string(settings.step.concurrent) + ")";
    if (settings.vram_conversations < 1 || settings.vram_conversations > get_most_concurrent())
        return "conversations held in VRAM must be from 1 to " + std::to_string(get_most_concurrent());
    return {};
}

bool PagedEngine::trim_gpu_conversations(int maximum, const Held* keep, std::optional<int64_t> priority) {
    while (get_resident_conversations() > maximum) {
        Held* victim = find_idle(keep);
        if (!victim)
            victim = find_decoding_victim(keep, priority);
        if (!victim)
            return false;
        pause_for_capacity(*victim);
    }
    return true;
}

void PagedEngine::end_preempted(size_t position) {
    Held& h = get_held(position);
    finish_request(h, "preempted");
}

int64_t PagedEngine::take_conversation(size_t target, const std::vector<int64_t>& ids, uint64_t image_key,
                                         int64_t priority) {
    const auto get_start = [&](const std::vector<int32_t>& live, const std::vector<int64_t>& points, uint64_t key) {
        if (key != image_key)
            return int64_t{0};
        return (int64_t) strata::core::get_resume_tokens(live, points, ids);
    };
    size_t best = target;
    const Held& initial = get_held(target);
    int64_t most = get_start(initial.live, model_.get_rewind_points(initial.parts), initial.image_key);
    for (size_t i = 0; i < at_.size(); ++i) {
        const Held& h = get_held(i);
        if (const int64_t start = get_start(h.live, model_.get_rewind_points(h.parts), h.image_key);
            !h.decoding && start > most) {
            best = i;
            most = start;
        }
    }
    auto back = paused_.end();
    for (auto p = paused_.begin(); p != paused_.end(); ++p)
        if (const int64_t start = get_start(p->head.live, p->head.points, p->head.image_key); start > most) {
            back = p;
            most = start;
        }
    if (back == paused_.end())
        std::swap(at_[target], at_[best]);
    Held& h = get_held(target);
    std::string err;
    // Preserve independent branches; temporary turn tails can be replaced in place by models that identify them.
    if (back == paused_.end() && most > 0 && most < (int64_t) h.live.size() &&
        model_.keeps_rewind_branch(h.parts, most)) {
        if (!pause(h, err)) {
            say("pausing a conversation failed, so it goes back in place and is not kept whole: %s\n", err.c_str());
        } else {
            back = std::find_if(paused_.begin(), paused_.end(), [&](const Paused& x) { return x.serial == serial_; });
            if (back == paused_.end())   // trim_ssd forgot it at once
                return 0;
        }
    }
    const bool whole = most == 0 || most == (int64_t) (back == paused_.end() ? h.live : back->head.live).size();
    // out of the list first: pausing the conversation in its place changes the list
    std::optional<Paused> p;
    if (back != paused_.end()) {
        p = std::move(*back);
        paused_.erase(back);
    }
    if ((p || most == 0) && !h.live.empty())
        pause_or_forget(h);
    if (p) {
        const bool ok = resume(*p, h, most, err, priority);
        if (whole || !model_.keeps_rewind_branch(h.parts, most)) {
            drop_pages(*p);
        } else {   // it stays paused whole; the pauses that made room for it did not count its bytes
            paused_.push_back(std::move(*p));
            trim_ram();
            trim_ssd();
        }
        if (!ok) {
            say("resuming a paused conversation failed, so its prompt is read again: %s\n", err.c_str());
            h.forget();
            return 0;
        }
    }
    if (whole)
        return most;
    if (!model_.rewind(h.parts, most, err)) {
        say("going back to token %lld of a conversation failed, so its prompt is read again: %s\n", (long long) most,
            err.c_str());
        h.forget();
        return 0;
    }
    const auto differs = std::mismatch(h.live.begin(), h.live.end(), ids.begin(), ids.end()).first;
    say("a conversation of %zu tokens goes back to token %lld (the new prompt differs from token %lld)\n",
        h.live.size(), (long long) most, (long long) (differs - h.live.begin()));
    h.live.resize((size_t) most);
    h.gpu_tokens.shrink_to((uint64_t) most);
    for (const std::unique_ptr<PagedPart>& part : h.parts)
        part->truncate(most);
    return most;
}

bool PagedEngine::make_room(Held& h, int64_t tokens, std::string& err, std::optional<int64_t> priority) {
    const uint64_t before = h.gpu_tokens.get_used();
    while (!h.gpu_tokens.try_reserve((uint64_t) tokens)) {
        Held* victim = find_idle(&h);
        if (!victim && priority)
            victim = find_decoding_victim(&h, priority);
        if (!victim) {
            err = "the shared GPU conversation budget is " + std::to_string(settings_.tokens_max) + " tokens";
            return false;
        }
        pause_for_capacity(*victim);
    }
    const bool reuse = std::all_of(h.parts.begin(), h.parts.end(),
        [tokens](const auto& part) { return part->can_reserve_without_growth(tokens); });
    if (!reuse && !model_.wait_state(err)) {
        h.gpu_tokens.shrink_to(before);
        return false;
    }
    for (size_t k = 0; k < h.parts.size(); ++k) {
        const strata::core::OnDevice on(cards_[k].dev);
        const auto& pool = cards_[k].experts.pool();
        const uint64_t wanted = pool.pages_for(h.parts[k]->get_pool_tokens(tokens)) * pool.page_bytes();
        if (!reclaim_stale_gpu_state(cards_[k], wanted - std::min(wanted, h.parts[k]->bytes()), &h, err)) {
            h.gpu_tokens.shrink_to(before);
            return false;
        }
        while (!h.parts[k]->reserve(tokens, err)) {
            if (!model_.wait_state(err)) {
                h.gpu_tokens.shrink_to(before);
                return false;
            }
            Held* idle = find_idle(&h);
            if (idle == nullptr) {
                h.gpu_tokens.shrink_to(before);
                return false;
            }
            pause_or_forget(*idle);
        }
    }
    if (!publish_residency(err)) {
        h.gpu_tokens.shrink_to(before);
        return false;
    }
    return true;
}

Held* PagedEngine::find_decoding_victim(const Held* keep, std::optional<int64_t> priority) {
    Held* victim = nullptr;
    for (const auto& h : held_)
        if (h.get() != keep && h.get() != reading_ && h->decoding && h->position >= 0 &&
            (!priority || h->dec.priority < *priority) &&
            (!victim || h->dec.priority < victim->dec.priority ||
             (h->dec.priority == victim->dec.priority && h->used > victim->used)))
            victim = h.get();
    return victim;
}

void PagedEngine::pause_for_capacity(Held& h) {
    if (h.decoding)
        end_preempted((size_t) h.position);
    pause_or_forget(h);
}

Held* PagedEngine::find_idle(const Held* keep, const std::function<bool(const Held&)>& can_go) const {
    Held* idle = nullptr;
    for (const std::unique_ptr<Held>& x : held_)
        if (x.get() != keep && x.get() != reading_ && !x->decoding && x->is_resident() && (!can_go || can_go(*x)) &&
            (idle == nullptr || std::pair{!x->live.empty(), x->get_yield_order()} <
                                std::pair{!idle->live.empty(), idle->get_yield_order()}))
            idle = x.get();
    return idle;
}

void PagedEngine::pause_or_forget(Held& h) {
    std::string err;
    if (!h.live.empty() && pause(h, err))
        return;
    if (!err.empty())
        say("pausing a conversation failed, so it is forgotten: %s\n", err.c_str());
    h.release();
}

bool PagedEngine::init_pages(Held& h, std::string& err, std::optional<int64_t> priority) {
    if (!h.is_resident() && !trim_gpu_conversations((int) settings_.vram_conversations - 1, &h, priority)) {
        err = "the GPU conversation limit is occupied by active requests";
        return false;
    }
    for (PagedCard& card : cards_) {
        PagedPart& part = *h.parts[card.index];
        if (part.is_ready())   // an earlier call failed on a later card
            continue;
        const auto& pool = card.experts.pool();
        if (!reclaim_stale_gpu_state(card, pool.pages_for(0) * pool.page_bytes(), &h, err)) {
            return false;
        }
        const strata::core::OnDevice on(card.dev);
        if (!part.init(card.experts.pool(), (strata::core::ConversationId) h.session + 1, err)) {
            h.release();
            return false;
        }
    }
    return true;
}

bool PagedEngine::pause(Held& h, std::string& err, bool ram_only) {
    const Clock::time_point t0 = Clock::now();
    Paused p;
    p.serial = ++serial_;
    p.priority = h.dec.priority;
    p.used = h.used;
    p.policy = h.policy;
    p.head.live = h.live;
    p.head.last_tokens_at = h.last_tokens_at;
    p.head.image_key = h.image_key;
    p.head.points = model_.get_rewind_points(h.parts);
    if (!model_.save_state(h.parts, p.head.state, err))
        return false;
    strata::core::PagePool& ram = ram_.pages().pool();
    const int64_t tokens = h.parts[0]->get_pool_tokens((int64_t) h.live.size());
    p.bytes = ram.pages_for(tokens) * ram.page_bytes() + p.head.state.capacity();
    const uint64_t budget = ram_only ? UINT64_MAX : get_ram_pause_budget();
    while (p.bytes <= budget && get_paused_bytes(true) + p.bytes > budget)
        if (!move_down())
            break;
    const bool in_ram = get_paused_bytes(true) + p.bytes <= budget && ram.grow_conversation(p.serial, tokens).empty();
    if (in_ram) {
        ram.set_conversation_label(p.serial, "Paused conversation " + std::to_string(p.serial));
        ram.set_paused(p.serial, true);
    } else if (ram_only) {
        err = "the RAM's pool has no room for a conversation of " + std::to_string(h.live.size()) + " tokens";
        return false;
    } else {   // the file takes every block the cards' pools gave the conversation
        p.file = get_pause_file(p.serial);
        p.head.cut = get_cut();
        p.bytes = h.get_bytes();
    }
    bool ok = true;
    for (size_t k = 0; in_ram && ok && k < h.parts.size(); ++k) {
        const PagedCard& card = cards_[k];
        const strata::core::OnDevice on(card.dev);
        ok = strata::core::copy_conversation(card.experts.pool(), h.parts[k]->id(), ram, p.serial, err,
                                             model_.get_gpu_span(card), model_.get_ram_span(card.lb, card.le));
    }
    ok = in_ram ? ok : save_pause_file(p, p.file, get_gpu_sections(h), err);
    if (!ok) {
        drop_pages(p);
        return false;
    }
    if (!in_ram) {
        p.bytes = std::filesystem::file_size(p.file);
        std::vector<uint8_t>().swap(p.head.state);
    }
    say("a conversation of %zu tokens is paused %s (%.0f MiB in %.0f ms)\n", h.live.size(),
        p.file.empty() ? "in RAM" : "on the SSD", (double) p.bytes / 1048576.0, get_ms_since(t0));
    h.release();
    paused_.push_back(std::move(p));
    trim_ssd();
    return true;
}

bool PagedEngine::resume(const Paused& p, Held& h, int64_t tokens, std::string& err,
                         std::optional<int64_t> priority) {
    const Clock::time_point t0 = Clock::now();
    const std::vector<int32_t>& live = p.head.live;
    // the copies below take as many first blocks as `h` holds
    if ((!h.has_pages() && !init_pages(h, err, priority)) || !make_room(h, tokens, err, priority))
        return false;
    strata::core::PagePool& ram = ram_.pages().pool();
    strata::core::PauseStream file;
    strata::core::PauseHeader saved;
    if (!p.file.empty() &&
        (!find_identity(err) || open_pause_file(p.file, *identity_, (uint64_t) facts_.context, file, saved, err) !=
                                    strata::core::SessionError::none))
        return false;
    // sections of cards at another cut come together in the RAM's pool, whose layout has every layer
    const bool through_ram = !p.file.empty() && p.head.cut != get_cut();
    if (through_ram) {
        const int64_t whole = h.parts[0]->get_pool_tokens((int64_t) live.size());   // (sections are read whole)
        if (const std::string why = ram.grow_conversation(p.serial, whole); !why.empty()) {
            err = "the RAM's pool has no room for a conversation paused before the layer cut moved: " + why;
            return false;
        }
        for (size_t k = 0; k <= p.head.cut.size(); ++k) {
            const auto [lb, le] = get_card_layers(p.head.cut, k, facts_.n_layers);
            if (!strata::core::read_conversation_section(file.get(), ram, p.serial, err, model_.get_ram_span(lb, le))) {
                ram.remove_conversation(p.serial);
                return false;
            }
        }
    }
    bool ok = true;
    for (size_t k = 0; ok && k < h.parts.size(); ++k) {
        const PagedCard& card = cards_[k];
        const strata::core::ConversationId id = h.parts[k]->id();
        const strata::core::OnDevice on(card.dev);
        ok = p.file.empty() || through_ram
                 ? strata::core::copy_conversation(ram, p.serial, card.experts.pool(), id, err,
                                                   model_.get_ram_span(card.lb, card.le), model_.get_gpu_span(card))
                 : strata::core::read_conversation_section(file.get(), card.experts.pool(), id, err,
                                                           model_.get_gpu_span(card));
    }
    if (through_ram)
        ram.remove_conversation(p.serial);
    if (!ok || !model_.restore_state(p.file.empty() ? p.head.state : saved.state, h.parts, live, tokens, err))
        return false;
    h.policy = p.policy;
    h.live = live;
    h.last_tokens_at = p.head.last_tokens_at;
    h.image_key = p.head.image_key;
    say("a conversation of %zu tokens is back from %s (%.0f ms)\n", live.size(), p.file.empty() ? "RAM" : "the SSD",
        get_ms_since(t0));
    return true;
}

bool PagedEngine::move_down() {
    const auto p = find_lowest(true);
    if (p == paused_.end())
        return false;
    move_to_ssd(p);
    return true;
}

bool PagedEngine::move_to_ssd(std::vector<Paused>::iterator p) {
    std::string err;
    const std::string file = get_pause_file(p->serial);
    p->head.cut = get_cut();
    const bool ok = save_pause_file(*p, file, get_ram_sections(p->serial), err);
    drop_pages(*p);   // (the RAM's pool's: p->file is empty yet)
    if (ok) {
        p->file = file;
        p->bytes = std::filesystem::file_size(file);
        std::vector<uint8_t>().swap(p->head.state);
        trim_ssd();
        return true;
    }
    say("moving a paused conversation to the SSD failed, so it is forgotten: %s\n", err.c_str());
    paused_.erase(p);
    return false;
}

void PagedEngine::trim_ram() {
    while (get_paused_bytes(true) > get_ram_pause_budget() && move_down()) {
    }
}

void PagedEngine::trim_ssd() {
    const uint64_t budget = (uint64_t) settings_.ssd_pause_mib << 20;
    while (get_paused_bytes(false) > budget) {
        const auto p = find_lowest(false);
        say("a paused conversation of %zu tokens is forgotten (--ssd-pause-mib %lld)\n", p->head.live.size(),
            (long long) settings_.ssd_pause_mib);
        drop_pages(*p);
        paused_.erase(p);
    }
}

void PagedEngine::drop_pages(const Paused& p) {
    std::error_code ec;
    if (p.file.empty())
        ram_.pages().pool().remove_conversation(p.serial);
    else
        std::filesystem::remove(p.file, ec);
}

uint64_t PagedEngine::get_paused_bytes(bool in_ram) const {
    uint64_t bytes = 0;
    for (const Paused& p : paused_)
        if (p.file.empty() == in_ram)
            bytes += p.bytes;
    return bytes;
}

uint64_t PagedEngine::get_ram_pause_budget() const {
    const strata::core::PagePool& ram = ram_.pages().pool();
    return ram.pages_for(settings_.tokens_max) * ram.page_bytes();
}

std::string PagedEngine::get_pause_file(uint64_t serial) const {
    return pause_prefix_ + std::to_string(serial) + ".bin";
}

std::vector<strata::core::PauseSection> PagedEngine::get_gpu_sections(const Held& h) const {
    std::vector<strata::core::PauseSection> sections;
    for (const PagedCard& card : cards_)
        sections.push_back({&card.experts.pool(), h.parts[card.index]->id(), model_.get_gpu_span(card), card.dev});
    return sections;
}

std::vector<strata::core::PauseSection> PagedEngine::get_ram_sections(uint64_t serial) const {
    std::vector<strata::core::PauseSection> sections;
    for (const PagedCard& card : cards_)
        sections.push_back({&ram_.pages().pool(), serial, model_.get_ram_span(card.lb, card.le)});
    return sections;
}

bool PagedEngine::save_pause_file(const Paused& p, const std::string& path,
                                  const std::vector<strata::core::PauseSection>& sections, std::string& err) {
    return find_identity(err) && strata::core::write_pause_file(path, *identity_, p.head, sections, err);
}

std::vector<Paused>::iterator PagedEngine::find_lowest(bool in_ram) {
    auto lowest = paused_.end();
    for (auto p = paused_.begin(); p != paused_.end(); ++p)
        if (p->file.empty() == in_ram &&
            (lowest == paused_.end() ||
             std::make_pair(p->priority, p->used) < std::make_pair(lowest->priority, lowest->used)))
            lowest = p;
    return lowest;
}

bool PagedEngine::adapt_experts() {
    std::vector<float>& usage = model_.get_usage();
    if (usage.empty() || steps_ % model_.get_adapt_interval(steps_) != 0)
        return true;
    const Clock::time_point t0 = Clock::now();
    std::vector<int32_t>& residency = model_.get_residency();
    const int64_t n_expert = facts_.n_expert;
    seen_.resize(usage.size(), 0.0f);
    for (size_t i = 0; i < usage.size(); ++i)
        if (const int64_t l = (int64_t) i / n_expert, e = (int64_t) i % n_expert; usage[i] > seen_[i]) {
            PagedCard* const holder = residency[i] >= 0 ? &get_card_of(l)
                                 : is_lent(l, e)     ? &cards_[(size_t) holder_[(size_t) l]]
                                                     : nullptr;
            if (holder != nullptr)
                holder->experts.note_use({(int32_t) l, (int32_t) e});
        }
    std::vector<strata::core::ExpertSwap> swaps = strata::core::choose_expert_swaps(
        usage, residency, facts_.n_layers, n_expert, facts_.adapt_min_gain,
        [this](int64_t l, int32_t e) { return is_lent(l, e); });
    if (swaps.size() > (size_t) facts_.adapt_swaps)
        swaps.resize((size_t) facts_.adapt_swaps);
    std::string err;
    for (const strata::core::ExpertSwap& s : swaps)
        if (!get_card_of(s.layer).experts.replace(s.layer, s.out, s.in, err)) {
            print_response("ERR moving an expert into the GPU's pool: %s\n", err.c_str());
            return false;
        }
    for (float& u : usage)
        u *= facts_.adapt_decay;
    seen_ = usage;
    if (!publish_residency(err)) {
        print_response("ERR %s\n", err.c_str());
        return false;
    }
    swapped_ += (int64_t) swaps.size();
    adapt_ms_ += get_ms_since(t0);
    return true;
}

bool PagedEngine::refill_experts() {
    std::string err;
    uint64_t wait_ms = 0;
    int loaded = 0;
    for (PagedCard& card : cards_) {
        uint64_t card_wait_ms = 0;
        if (card.experts.is_refill_pending() && !card.experts.is_refill_wanted())
            if (Held* stale = find_stale()) {
                if (!wait_model_state())
                    return false;
                pause_or_forget(*stale);
            }
        const int got = card.experts.is_refill_wanted() ? card.experts.refill(card.refill_batch, card_wait_ms, err) : 0;
        loaded = got < 0 ? -1 : loaded + got;
        if (card_wait_ms > 0 && (wait_ms == 0 || card_wait_ms < wait_ms))
            wait_ms = card_wait_ms;
        if (loaded < 0)
            break;
    }
    if (loaded == 0 && ram_.pages().is_refill_wanted()) {   // the GPUs' first: their experts are the hotter ones
        uint64_t ram_wait_ms = 0;
        loaded = ram_.pages().refill(kRefillBatch, ram_wait_ms, err);
        if (ram_wait_ms > 0 && (wait_ms == 0 || ram_wait_ms < wait_ms))
            wait_ms = ram_wait_ms;
    }
    if (loaded < 0 || !publish_residency(err)) {
        say("putting experts back in their pools: %s\n", err.c_str());
        return false;
    }
    if (wait_ms > 0)
        input_.wait_for_line(std::chrono::milliseconds(wait_ms), [] { return false; });
    return true;
}

bool PagedEngine::is_refill_wanted() const {
    if (ram_.pages().is_refill_wanted() ||
        std::any_of(cards_.begin(), cards_.end(), [](const auto& card) { return card.experts.is_refill_wanted(); }))
        return true;
    return find_stale() != nullptr && std::any_of(cards_.begin(), cards_.end(), [](const auto& card) {
        return card.experts.is_refill_pending();
    });
}

bool PagedEngine::publish_residency(std::string& err) {
    if (!residency_changed_)
        return true;
    for (const PagedCard& card : cards_)
        if (!model_.publish_residency(card, err))
            return false;
    residency_changed_ = false;
    return true;
}

void PagedEngine::poll_cancellations() {
    for (const auto& h : held_)
        if (h->decoding)
            h->dec.stop |= input_.is_cancel_requested(h->request_id);
}

bool PagedEngine::is_decoding() const {
    return std::any_of(held_.begin(), held_.end(), [](const std::unique_ptr<Held>& h) { return h->decoding; });
}

std::vector<strata::core::HeldConversation> PagedEngine::list_conversations() const {
    std::vector<strata::core::HeldConversation> list;
    for (size_t i = 0; i < at_.size(); ++i) {
        const size_t place = i;
        const Held& h = get_held(place);
        if (h.live.empty() && !h.decoding && &h != reading_)
            continue;
        strata::core::HeldConversation c;
        c.id = "conversation" + std::to_string(h.session);
        c.state = &h == reading_ ? "reading" : h.decoding ? "decoding" : h.live.empty() ? "empty" : "holding";
        c.ids = h.live.data();
        c.n = h.live.size();
        c.capacity = (uint64_t) settings_.conversation_context;
        c.bytes = c.used = h.get_bytes();
        c.last_tokens_at = h.last_tokens_at;
        list.push_back(c);
    }
    for (const Paused& p : paused_) {
        strata::core::HeldConversation c;
        c.id = "parked" + std::to_string(p.serial);
        c.place = p.file.empty() ? "ram" : "disk";
        c.state = "parked";
        c.ids = p.head.live.data();
        c.n = c.capacity = p.head.live.size();
        c.bytes = c.used = p.bytes;
        c.last_tokens_at = p.head.last_tokens_at;
        list.push_back(c);
    }
    return list;
}

strata::serve::MemoryReport PagedEngine::describe_memory() const {
    const strata::core::PagePoolReport rr = ram_.pages().pool().report();
    strata::serve::MemoryReport report;
    report.pause_directory = std::filesystem::absolute(pause_prefix_).parent_path().string();
    for (const PagedCard& card : cards_) {
        const strata::core::PagePoolReport r = card.experts.pool().report();
        strata::serve::Memory& gpu = report.at(card.id);
        strata::core::read_gpu_memory(card.dev, gpu.name, gpu.total, gpu.free, gpu.unified);
        gpu.add_blocks(card.experts.pool().get_memory_blocks());
        gpu.add_blocks(strata::core::get_memory_manager().get_blocks(card.id));
        gpu.pool_pages = r.total.pages;
        gpu.allocated_pages = r.allocated.pages;
        gpu.fragmented_pages = r.fragmented.pages;
        gpu.setting = r.total.bytes;
        gpu.smallest = get_smallest_gpu(card, settings_.tokens_min);
        gpu.largest = get_largest_gpu(card);
    }
    strata::serve::Memory& ram = report.at("ram");
    ram.add_blocks(ram_.pages().pool().get_memory_blocks());
    ram.add_blocks(strata::core::get_memory_manager().get_blocks("ram"));
    for (size_t i = 0; i < at_.size(); ++i) {
        const auto& h = get_held(i);
        const std::string id = "conversation" + std::to_string(h.session);
        ram.add_blocks({{{"Checkpoints", "Host checkpoints", "Conversation " + id},
                         model_.get_host_state_bytes(h.parts)}});
    }
    for (const auto& p : paused_)
        ram.add_blocks({{{"Checkpoints", "Host checkpoints", "Paused conversation " + std::to_string(p.serial)},
                         strata::core::get_owned_vector_bytes(p.head.state)}});
    ram.paused_budget = get_ram_pause_budget();
    ram.pool_pages = rr.total.pages;
    ram.allocated_pages = rr.allocated.pages;
    ram.fragmented_pages = rr.fragmented.pages;
    ram.setting = rr.total.bytes + ram_.ring_bytes();
    ram.mapped = strata::core::get_memory_manager().get_mapped_bytes();
    ram.smallest = get_smallest_ram();
    ram.largest = get_largest_ram();
    strata::serve::Memory& ssd = report.at("ssd");
    for (int64_t l = 0; l < facts_.n_layers; ++l) {
        uint64_t on_ssd = 0;
        for (int64_t e = 0; e < facts_.n_expert; ++e)
            if (model_.get_residency()[(size_t) (l * facts_.n_expert + e)] < 0 && !is_lent(l, e) && !ram_.holds(l, e))
                on_ssd += facts_.expert_bytes[(size_t) l];
        ssd.add_blocks({{{"experts", "Model files",
            strata::core::get_expert_memory_label((int32_t) l, facts_.expert_bytes[(size_t) l])}, on_ssd}});
    }
    ssd.paused_budget = (uint64_t) settings_.ssd_pause_mib << 20;
    const int64_t probe = std::max<int64_t>(facts_.context, 1);   // tokens whose pages give the average
    uint64_t fixed = 0, grown = 0;   // a conversation's pages on every card: at no token, at `probe` tokens
    for (const PagedCard& card : cards_) {
        const strata::core::PagePool& pool = card.experts.pool();
        fixed += pool.pages_for(0) * pool.page_bytes();
        grown += pool.pages_for(probe) * pool.page_bytes();
    }
    report.settings = {{"context", settings_.conversation_context},
                       {"context_capacity", facts_.context},
                       {"concurrent", settings_.step.concurrent},
                       {"vram_conversations", settings_.vram_conversations},
                       {"conversation_idle_seconds", settings_.conversation_idle_seconds},
                       {"decode_share_percent", settings_.decode_share_percent},
                       {"prompt_turn_ms", settings_.prompt_turn_ms},
                       {"slots", get_most_concurrent()},
                       {"rows", settings_.step.rows},
                       {"lookahead", settings_.step.lookahead},
                       {"tokens_min", settings_.tokens_min},
                       {"tokens_max", settings_.tokens_max},
                       {"ssd_pause_mib", settings_.ssd_pause_mib}};
    for (const PagedCard& card : cards_)
        report.settings.emplace_back("memory_mib." + card.id,
                                     (int64_t) (card.experts.pool().report().total.bytes >> 20));
    report.settings.insert(report.settings.end(),
                           {{"memory_mib.ram", (int64_t) ((rr.total.bytes + ram_.ring_bytes()) >> 20)},
                            {"token_bytes", (int64_t) ((grown - fixed) / probe)},
                            {"conversation_bytes", (int64_t) fixed}});
    report.settings.emplace_back("drafter", settings_.drafter && facts_.has_drafter ? 1 : 0);
    report.settings.emplace_back("save_on_shutdown", settings_.save_on_shutdown ? 1 : 0);
    report.settings.emplace_back("suffix_draft", settings_.suffix_draft);
    report.settings.emplace_back("lookup_chain", settings_.lookup_chain);
    uint64_t drafter_bytes = facts_.drafter_expert_bytes;
    for (const PagedCard& card : cards_)
        drafter_bytes += model_.get_drafter_weight_bytes(card);
    char cost[32];
    std::snprintf(cost, sizeof cost, "%.1f GiB", (double) drafter_bytes / 1073741824.0);
    report.details = {{"drafter_label", "Use " + facts_.drafter_label + " drafter (" + cost + ")"},
                      {"drafter_details", facts_.has_drafter ? "Discovered drafter: " + facts_.drafter_label :
                       "No model drafter was loaded on startup; restart with " + facts_.drafter_start},
                      {"suffix_label", "Use suffix drafter (0 GiB)"},
                      {"kv_quantization", facts_.kv_quantization}};
    report.settings.emplace_back("has_drafter", facts_.has_drafter ? 1 : 0);
    report.per_conversation = true;
    return report;
}

std::string PagedEngine::change_settings(const SettingValues& values, std::string& shortfall) {
    PagedSettings s = settings_;
    std::vector<std::optional<uint64_t>> gpu_bytes(cards_.size());   // per card
    std::optional<uint64_t> ram_bytes;
    for (const auto& [name, text] : values) {
        if (name == "drafter" || name == "save_on_shutdown") {
            if (text != "on" && text != "off")
                return name + " takes on or off, not " + text;
            (name == "drafter" ? s.drafter : s.save_on_shutdown) = text == "on";
            continue;
        }
        char* end = nullptr;
        const long long v = std::strtoll(text.c_str(), &end, 10);
        if (text.empty() || *end != '\0' || v < 0 || v > INT32_MAX)
            return name + " takes a whole number from 0 to " + std::to_string(INT32_MAX) + ", not " + text;
        const auto card = std::find_if(cards_.begin(), cards_.end(),
                                       [&](const PagedCard& c) { return name == "memory_mib." + c.id; });
        if (card != cards_.end())
            gpu_bytes[card->index] = round_to_chunks(card->experts.pool(), (uint64_t) v << 20);
        else if (name == "memory_mib.ram")
            ram_bytes = round_to_chunks(ram_.pages().pool(), std::max((uint64_t) v << 20, ram_.ring_bytes()) -
                                                              ram_.ring_bytes()) + ram_.ring_bytes();
        else if (name == "context")
            s.conversation_context = v;
        else if (name == "suffix_draft")
            s.suffix_draft = v;
        else if (name == "lookup_chain")
            s.lookup_chain = v;
        else if (name == "tokens_min")
            s.tokens_min = v;
        else if (name == "tokens_max")
            s.tokens_max = v;
        else if (name == "ssd_pause_mib")
            s.ssd_pause_mib = v;
        else if (name == "concurrent")
            s.step.concurrent = (int) v;
        else if (name == "vram_conversations")
            s.vram_conversations = v;
        else if (name == "conversation_idle_seconds")
            s.conversation_idle_seconds = v;
        else if (name == "decode_share_percent")
            s.decode_share_percent = v;
        else if (name == "prompt_turn_ms")
            s.prompt_turn_ms = v;
        else if (name == "rows")
            s.step.rows = (int) v;
        else if (name == "lookahead")
            s.step.lookahead = (int) v;
        else
            return facts_.name + " has no setting " + name;
    }
    // every check first, so that a refusal changes nothing
    const auto gib = [](uint64_t b) {
        char text[32];
        std::snprintf(text, sizeof text, "%.2f GiB", (double) b / 1073741824.0);
        return std::string(text);
    };
    const int slots = get_most_concurrent();
    if (s.step.concurrent < 1 || s.step.concurrent > slots)
        return "the conversations at once may be 1 to " + std::to_string(slots) +
               " (the most one step of this model runs)";
    if (const std::string why = get_step_settings_problem(s.step, facts_.max_rows); !why.empty())
        return why;
    if (const std::string why = get_conversation_count_error(s); !why.empty())
        return why;
    if (s.drafter && !facts_.has_drafter)
        return "this engine has no drafter: it needs " + facts_.drafter_start + " at its start";
    if (const std::string why = strata::serve::get_token_limits_error(s.conversation_context, facts_.context,
            s.tokens_min, s.tokens_max); !why.empty())
        return why;
    for (const std::unique_ptr<Held>& h : held_)
        if (h->decoding && (int64_t) h->live.size() + 1 >= s.conversation_context)
            return "a conversation decoding now holds " + std::to_string(h->live.size()) +
                   " tokens and needs room for its next token; the per-conversation limit must exceed " +
                   std::to_string(h->live.size() + 1);
    std::vector<uint64_t> gpu_now;
    for (const PagedCard& card : cards_) {
        const uint64_t now = gpu_now.emplace_back(card.experts.pool().report().total.bytes);
        const uint64_t smallest = get_smallest_gpu(card, s.tokens_min), asked = gpu_bytes[card.index].value_or(now);
        if (asked < smallest)
            return "the biggest active conversation and the token minimum need " + gib(smallest) + "; the pool of " +
                   card.id + " would keep " + gib(asked);
        if (gpu_bytes[card.index] && asked > get_largest_gpu(card))
            return card.id + " has room for " + gib(get_largest_gpu(card)) + " of pool, not " + gib(asked);
    }
    // (each range takes all of get_commit_room: on the Strix Halo, where the GPU's memory is commit too, one change
    // that grows both pools must share the remaining commit capacity)
    if (ram_bytes && (*ram_bytes < get_smallest_ram() || *ram_bytes > get_largest_ram()))
        return "the RAM's pool may be " + gib(get_smallest_ram()) + " to " + gib(get_largest_ram()) + ", not " +
               gib(*ram_bytes);
    // then the changes: those that free memory first, so that those taking memory find it; a higher token minimum
    // waits for the GPU's pool to grow, as the pool may not hold it before
    const int64_t min_first = std::min(s.tokens_min, settings_.tokens_min);
    if (const std::string why = set_token_limits(min_first, s.tokens_max, s.conversation_context); !why.empty())
        return why;
    (void) trim_gpu_conversations((int) s.vram_conversations);
    settings_.vram_conversations = s.vram_conversations;
    settings_.conversation_idle_seconds = s.conversation_idle_seconds;
    settings_.decode_share_percent = s.decode_share_percent;
    settings_.prompt_turn_ms = s.prompt_turn_ms;
    settings_.save_on_shutdown = s.save_on_shutdown;
    std::string err;
    if (!s.drafter && settings_.drafter)
        (void) switch_drafter(false, err);
    settings_.ssd_pause_mib = s.ssd_pause_mib;
    trim_ssd();
    settings_.step = s.step;
    settings_.suffix_draft = s.suffix_draft;
    settings_.lookup_chain = s.lookup_chain;
    const auto note = [&shortfall](const char* what, const std::string& why) {
        if (!why.empty())
            shortfall += (shortfall.empty() ? "" : "; ") + (what + why);
    };
    const uint64_t ram_now = ram_.pages().pool().report().total.bytes + ram_.ring_bytes();
    const auto resize = [&](bool shrink) {
        for (PagedCard& card : cards_)
            if (const std::optional<uint64_t> asked = gpu_bytes[card.index];
                asked && *asked != gpu_now[card.index] && (*asked < gpu_now[card.index]) == shrink)
                note(("the pool of " + card.id + ": ").c_str(), resize_gpu(card, *asked));
        if (ram_bytes && *ram_bytes != ram_now && (*ram_bytes < ram_now) == shrink)
            note("the RAM's pool: ", resize_ram(*ram_bytes));
    };
    resize(true);
    if (s.drafter && !settings_.drafter && !switch_drafter(true, err))
        note("", err);
    resize(false);
    if (std::any_of(gpu_bytes.begin(), gpu_bytes.end(), [](const auto& b) { return b.has_value(); })) {
        const std::vector<int64_t> cut = plan_cut();
        const bool move = cut != get_cut();
        if (move && !move_cut(cut, err))
            note("moving the layer cut: ", err);
        if ((!move || cut != get_cut()) && !holder_.empty()) {   // (a cut move places them for its new cut)
            place_card_experts(true);
            refilling_ = true;
        }
    }
    if (s.tokens_min != min_first)
        note("", set_token_limits(s.tokens_min, s.tokens_max, s.conversation_context));
    apply_step_settings();
    if (!shortfall.empty())
        say("CHANGE_SETTINGS changed less than asked: %s\n", shortfall.c_str());
    return {};
}

std::string PagedEngine::resize_gpu(PagedCard& card, uint64_t bytes) {
    const strata::core::OnDevice on(card.dev);
    strata::core::PagePool& pool = card.experts.pool();
    const Clock::time_point t0 = Clock::now();
    const strata::core::PagePoolReport before = pool.report();
    const uint64_t keep = bytes / pool.chunk_bytes() * pool.chunk_bytes() / pool.page_bytes();   // its pages after
    // the conversations that do not decode and would not fit in the smaller pool at all
    while (Held* h = find_idle(nullptr, [&](const Held& x) {
               return x.parts[card.index]->bytes() / pool.page_bytes() > keep;
           }))
        pause_or_forget(*h);
    for (;;) {
        const strata::core::PoolResize r = card.experts.resize(bytes);
        if (r.refusal.empty())
            break;
        if (r.to_pause.empty())
            return r.refusal;
        const auto in_the_way = [&](const Held& x) {
            return std::find(r.to_pause.begin(), r.to_pause.end(), (strata::core::ConversationId) x.session + 1) !=
                   r.to_pause.end();
        };
        // pausing any conversation makes room for the ones in the way: those that do not decode first
        if (Held* h = find_idle(nullptr, in_the_way); h != nullptr || (h = find_idle(nullptr)) != nullptr) {
            pause_or_forget(*h);
            continue;
        }
        Held* victim = nullptr;
        for (size_t i = 0; i < at_.size(); ++i)
            if (Held& h = get_held(i);
                h.decoding && in_the_way(h) && (victim == nullptr || h.get_yield_order() < victim->get_yield_order()))
                victim = &h;
        if (victim == nullptr)
            return r.refusal;
        end_preempted((size_t) victim->position);
        pause_or_forget(*victim);
    }
    refilling_ = refilling_ || pool.report().total.pages > before.total.pages;
    std::string err;
    for (const std::unique_ptr<Held>& h : held_)   // a shrink moved conversation blocks
        if (PagedPart& part = *h->parts[card.index]; part.is_ready() && !part.refresh(err))
            return err;
    if (!publish_residency(err))
        return err;
    report_resize(card.id.c_str(), before, pool.report(), t0);
    return {};
}

std::string PagedEngine::resize_ram(uint64_t bytes) {
    const Clock::time_point t0 = Clock::now();
    const strata::core::PagePoolReport before = ram_.pages().pool().report();
    for (;;) {
        const strata::core::PoolResize r = ram_.resize(bytes);
        if (r.refusal.empty())
            break;
        const auto p = std::find_if(paused_.begin(), paused_.end(), [&](const Paused& x) {
            return x.file.empty() &&
                   std::find(r.to_pause.begin(), r.to_pause.end(), (strata::core::ConversationId) x.serial) !=
                       r.to_pause.end();
        });
        if (p == paused_.end())
            return r.refusal;
        move_to_ssd(p);
    }
    refilling_ = refilling_ || ram_.pages().pool().report().total.pages > before.total.pages;
    trim_ram();
    report_resize("RAM", before, ram_.pages().pool().report(), t0);
    return {};
}

/// Says on stderr how the pool of memory `name` changed, from `before` to `after`, in the time since `t0`.
void PagedEngine::report_resize(const char* name, const strata::core::PagePoolReport& before,
                                const strata::core::PagePoolReport& after, Clock::time_point t0) const {
    const auto gib = [](const strata::core::PagePoolReport::Use& u) { return (double) u.bytes / 1073741824.0; };
    say("the %s's pool: %.2f -> %.2f GiB in %.0f ms (experts %.2f -> %.2f GiB)\n", name, gib(before.total),
        gib(after.total), get_ms_since(t0), gib(before.experts), gib(after.experts));
}

std::vector<int64_t> PagedEngine::plan_cut() const {
    using namespace strata::core;
    if (cards_.size() < 2)
        return {};
    const LayerSizes sizes = model_.get_layer_sizes(cards_[0].experts.pool().page_bytes());
    int64_t decoding = 0, tokens = settings_.tokens_min;   // the cards keep room for these conversations' state
    for (const std::unique_ptr<Held>& h : held_)
        if (h->decoding) {
            ++decoding;
            tokens = std::max(tokens, h->dec.pos + h->dec.max_new - h->dec.produced + settings_.step.lookahead + 1);
        }
    std::vector<uint64_t> pools;
    std::vector<double> links;
    for (const PagedCard& card : cards_) {
        pools.push_back(card.experts.pool().report().total.bytes);
        links.push_back(card.link.to_gbps);
    }
    SplitProblem problem = make_paged_split_problem(sizes, pools, std::max<int64_t>(decoding, 1),
                                                    std::min(tokens, facts_.context));
    const std::vector<double> miss_scale = strata::program::split_rules::link_miss_scale(links);
    for (size_t k = 0; k < cards_.size(); ++k)
        problem.cards[k] = {cards_[k].layer_ms, miss_scale[k]};
    for (const ExpertKey& key : facts_.split_ranking)
        problem.ranked_layers.push_back(key.layer);
    problem.can_start = [&](const std::vector<SplitFill>& fills) {
        for (size_t k = 0; k < fills.size(); ++k) {
            const int64_t lb = fills[k].lb, le = fills[k].le;
            if (lb < cards_[k].weights_lb || le > cards_[k].weights_le || !model_.can_run_layers(lb, le))
                return false;
        }
        return true;
    };
    const SplitPlan plan = plan_layer_split(problem, (facts_.name + ": the layer cut: ").c_str());
    return plan.cuts.empty() || plan.start_unchecked ? get_cut() : plan.cuts;
}

bool PagedEngine::move_cut(const std::vector<int64_t>& cut, std::string& err) {
    using namespace strata::core;
    const Clock::time_point t0 = Clock::now();
    const std::vector<int64_t> before = get_cut();
    std::vector<std::pair<Held*, Paused>> out;   // the conversations paused here
    // puts back those that decode (`all`: every one), the others staying paused
    const auto bring_back = [&](bool all) {
        for (auto& [h, p] : out) {
            if (!all && !h->decoding) {
                paused_.push_back(std::move(p));
                continue;
            }
            std::string why;
            const bool ok = resume(p, *h, (int64_t) p.head.live.size(), why);
            drop_pages(p);
            if (ok)
                continue;
            say("a conversation could not come back after the layer cut moved, so it is forgotten: %s\n",
                why.c_str());
            h->forget();
            if (h->decoding)
                end_preempted((size_t) h->position);
        }
    };
    for (const std::unique_ptr<Held>& h : held_) {
        if (h->live.empty()) {
            for (const std::unique_ptr<PagedPart>& part : h->parts)
                if (part->is_ready())
                    part->release();
            continue;
        }
        if (!pause(*h, err, true)) {
            bring_back(true);
            return false;
        }
        out.emplace_back(h.get(), std::move(paused_.back()));
        paused_.pop_back();
    }
    std::vector<StateLayout> old_layouts;
    for (size_t k = 0; k < cards_.size(); ++k) {
        PagePool& pool = cards_[k].experts.pool();
        const auto [lb, le] = get_card_layers(cut, k, facts_.n_layers);
        old_layouts.push_back(pool.layout());
        StateLayout layout;
        std::string why;
        if (!model_.get_card_layout(cards_[k], lb, le, layout, why) || !(why = pool.set_layout(layout)).empty()) {
            err = cards_[k].id + "'s pool: " + why;
            for (size_t j = 0; j < k; ++j)
                (void) cards_[j].experts.pool().set_layout(old_layouts[j]);
            bring_back(true);
            return false;
        }
    }
    // the experts lent to other cards leave first: one of a layer its holder takes would stay in `lent`, where the
    // holder's own windows do not look
    if (!holder_.empty())
        place_card_experts(false);
    if (!set_layers(cut, err)) {
        const std::string failure = err;
        for (size_t k = 0; k < cards_.size(); ++k)
            (void) cards_[k].experts.pool().set_layout(old_layouts[k]);
        std::string rollback_error;
        if (!set_layers(before, rollback_error)) {
            err = failure + "; restoring the old layer owners failed: " + rollback_error;
            fatal_error_ = err;
            for (auto& [h, p] : out)
                drop_pages(p);
            return false;
        }
        place_card_experts(false);
        if (!holder_.empty())
            place_card_experts(true);
        residency_changed_ = refilling_ = true;
        bring_back(true);
        err = failure;
        return false;
    }
    // a moved layer's experts leave the card it left, even those it is to hold for the layer's new card: the hook put
    // them in the device's table, not in `lent`
    place_card_experts(false);
    if (!holder_.empty())
        place_card_experts(true);
    residency_changed_ = refilling_ = true;
    bring_back(false);
    trim_ram();   // the conversations left paused went to RAM beyond --tokens-max's budget
    if (!fill_cards(err))
        return false;
    std::string layers;
    for (const PagedCard& card : cards_)
        layers += (layers.empty() ? "" : ", ") + card.id + " layers " + std::to_string(card.lb) + "-" +
                  std::to_string(card.le - 1);
    say("the layer cut moved: K=%s -> K=%s (%s; %zu conversations paused for it) in %.0f ms\n",
        format_cuts(before).c_str(), format_cuts(cut).c_str(), layers.c_str(), out.size(), get_ms_since(t0));
    return true;
}

bool PagedEngine::set_layers(const std::vector<int64_t>& cut, std::string& err) {
    for (size_t k = 0; k < cards_.size(); ++k) {
        PagedCard& card = cards_[k];
        std::tie(card.lb, card.le) = get_card_layers(cut, k, facts_.n_layers);
        if (!model_.set_card_layers(card, err))
            return false;
        for (const std::unique_ptr<Held>& h : held_)
            h->parts[k]->set_layers(card.lb, card.le);
    }
    return true;
}

bool PagedEngine::fill_cards(std::string& err) {
    for (PagedCard& card : cards_)
        while (card.experts.is_refill_wanted()) {
            uint64_t wait_ms = 0;
            if (card.experts.refill(card.refill_batch, wait_ms, err) < 0)
                return false;
            if (wait_ms > 0)
                std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms));
        }
    return publish_residency(err);
}

void PagedEngine::place_card_experts(bool lend) {
    using namespace strata::core;
    const int64_t n_layers = facts_.n_layers, n_expert = facts_.n_expert;
    ExpertPlacement placed;
    const std::vector<ExpertKey>& first = facts_.split_ranking;
    if (lend && !holder_.empty()) {
        const uint64_t page = cards_[0].experts.pool().page_bytes();
        std::vector<CardPool> pools;   // each card's pages less those kept for --tokens-min
        for (const PagedCard& card : cards_) {
            const PagePool& pool = card.experts.pool();
            const uint64_t pages = pool.report().total.pages;
            pools.push_back({card.lb, card.le, (pages - std::min(pages, pool.pages_for(settings_.tokens_min))) * page});
        }
        std::vector<uint64_t> bytes;
        for (const uint64_t b : facts_.expert_bytes)
            bytes.push_back((b + page - 1) / page * page);
        placed = place_experts(pools, first, n_layers, n_expert, bytes,
                               [&](int, int holder) { return is_hop_wanted(cards_[(size_t) holder]); });
    }
    if (!holder_.empty())   // in place: the model's hop holders point into it
        for (int64_t l = 0; l < n_layers; ++l)
            holder_[(size_t) l] = placed.remote.empty() ? -1 : placed.remote[(size_t) l];
    std::string lenders;
    for (PagedCard& card : cards_) {
        int64_t max_experts = facts_.gpu_experts;
        std::vector<ExpertKey> ranking = model_.get_card_ranking(card.lb, card.le, max_experts);
        if (!placed.card.empty()) {   // those placed on it, hottest first, then the rest of its layers'
            const auto is_on_card = [&](const ExpertKey& k) {
                return placed.card[(size_t) (k.layer * n_expert + k.expert)] == (int32_t) card.index;
            };
            std::vector<ExpertKey> held;
            std::copy_if(first.begin(), first.end(), std::back_inserter(held), is_on_card);
            max_experts = (int64_t) held.size();
            const int64_t others = std::count_if(held.begin(), held.end(), [&](const ExpertKey& k) {
                return k.layer < card.lb || k.layer >= card.le;
            });
            std::remove_copy_if(ranking.begin(), ranking.end(), std::back_inserter(held), is_on_card);
            ranking = std::move(held);
            if (others > 0)
                lenders += (lenders.empty() ? "" : ", ") + card.id + " " + std::to_string(others);
        }
        const OnDevice on(card.dev);
        card.experts.set_ranking(ranking, max_experts);
    }
    if (!lenders.empty())
        say("experts of other cards' layers, computed where they are in one hop: %s\n", lenders.c_str());
}

bool PagedEngine::is_hop_wanted(const PagedCard& holder) const {
    const uint64_t row = facts_.row_bytes;
    return settings_.remote_experts == "on" ||
           strata::core::is_hop_cheaper(strata::core::get_hop_ms(holder.link, settings_.step.rows, row, row), 1,
                                        facts_.cpu_expert_ms);
}

std::vector<int64_t> PagedEngine::get_cut() const {
    std::vector<int64_t> cut;
    for (size_t k = 1; k < cards_.size(); ++k)
        cut.push_back(cards_[k].lb);
    return cut;
}

std::string PagedEngine::set_token_limits(int64_t min, int64_t max, int64_t context) {
    while (gpu_token_budget_.get_used() > (uint64_t) max) {
        Held* victim = find_idle(nullptr);
        if (!victim)
            victim = find_decoding_victim();
        if (!victim)
            return "the GPU conversation budget cannot be reduced while its request is on the control channel";
        pause_for_capacity(*victim);
    }
    while (Held* h = find_idle(nullptr, [context](const Held& h) { return (int64_t) h.live.size() >= context; }))
        pause_or_forget(*h);
    // Pool spans include padding and spare KV cells; logical tokens share the engine's budget instead.
    for (PagedCard& card : cards_) {
        strata::core::PagePool& pool = card.experts.pool();
        if (const std::string why = pool.set_token_limits(min, pool.tokens_max()); !why.empty()) {
            for (size_t k = 0; k < card.index; ++k) {   // the cards before it go back to the old minimum
                strata::core::PagePool& done = cards_[k].experts.pool();
                (void) done.set_token_limits(settings_.tokens_min, done.tokens_max());
            }
            return why;
        }
    }
    (void) gpu_token_budget_.try_set_limit((uint64_t) max);
    for (const auto& h : held_)
        if (h->decoding)
            h->dec.max_new = std::min(h->dec.max_new, h->dec.produced +
                std::max<int64_t>(0, context - h->dec.pos - 1));
    settings_.conversation_context = context;
    settings_.tokens_min = min;
    settings_.tokens_max = max;
    trim_ram();
    return {};
}

bool PagedEngine::switch_drafter(bool on, std::string& err) {
    std::vector<uint64_t> targets, previous;
    for (const PagedCard& card : cards_) {
        const auto& pool = card.experts.pool();
        const uint64_t bytes = model_.get_drafter_weight_bytes(card) / pool.chunk_bytes() * pool.chunk_bytes();
        const uint64_t now = pool.report().total.bytes;
        previous.push_back(now);
        if (on && (bytes > now || now - bytes < get_smallest_gpu(card, settings_.tokens_min))) {
            err = "the pool cannot make room for the drafter's dense weights";
            return false;
        }
        targets.push_back(on ? now - bytes : now + bytes);
    }
    if (!on) {
        if (!model_.switch_drafter(false, err))
            return false;
        for (PagedCard& card : cards_)
            if (const std::string why = targets[card.index] == card.experts.pool().report().total.bytes
                    ? std::string{} : resize_gpu(card, targets[card.index]); !why.empty())
                say("the freed drafter memory could not all join the pool: %s\n", why.c_str());
        settings_.drafter = false;
        say("the drafter is off, its memory given back\n");
        return true;
    }
    for (PagedCard& card : cards_)
        if (const std::string why = targets[card.index] == card.experts.pool().report().total.bytes
                ? std::string{} : resize_gpu(card, targets[card.index]); !why.empty()) {
            err = why;
            return false;
        }
    if (!model_.switch_drafter(true, err)) {
        for (PagedCard& card : cards_)
            if (previous[card.index] != card.experts.pool().report().total.bytes)
                (void) resize_gpu(card, previous[card.index]);
        err = "the drafter could not be loaded: " + err;
        return false;
    }
    for (const std::unique_ptr<Held>& h : held_)
        model_.forget_drafts(h->session);
    settings_.drafter = true;
    say("the drafter is on\n");
    return true;
}

void PagedEngine::apply_step_settings() {
    step_.settings = settings_.step;
    const int stages = model_.get_pipeline_stages();
    const int concurrent = settings_.step.concurrent;
    int groups = settings_.batch_groups == "auto" ? std::min(stages, concurrent) :
                 std::atoi(settings_.batch_groups.c_str());
    if (settings_.batch_groups == "auto") {
        while (groups > 1 && concurrent % groups != 0)
            --groups;
    }
    step_.settings.groups = stages > 1 && groups > 1 && concurrent % groups == 0 ? groups : 1;
    step_.settings.max_draft_conversations = facts_.draft_limits.max_conversations;
    if (!is_model_drafting() && settings_.suffix_draft == 0 && settings_.lookup_chain == 0)
        step_.settings.lookahead = 0;
}

uint64_t PagedEngine::get_smallest_gpu(const PagedCard& card, int64_t tokens_min) const {
    const strata::core::PagePool& pool = card.experts.pool();
    uint64_t pages = pool.pages_for(tokens_min);
    for (const std::unique_ptr<Held>& h : held_)
        if (h->decoding) {
            const int64_t end = h->dec.pos + h->dec.max_new - h->dec.produced + settings_.step.lookahead + 1;
            const PagedPart& part = *h->parts[card.index];
            pages = std::max(pages, pool.pages_for(part.get_pool_tokens(std::min(end, facts_.context))));
        }
    const uint64_t chunk = pool.chunk_bytes();
    return std::max<uint64_t>((pages * pool.page_bytes() + chunk - 1) / chunk, 1) * chunk;
}

uint64_t PagedEngine::get_largest_gpu(const PagedCard& card) const {
    const strata::core::PagePool& pool = card.experts.pool();
    const uint64_t chunk = pool.chunk_bytes();
    return (pool.report().total.bytes + get_gpu_room(card.dev, model_.get_gpu_keep(card), ram_.get_mapped_bytes())) /
           chunk * chunk;
}

uint64_t PagedEngine::get_smallest_ram() const {
    return ram_.pages().pool().chunk_bytes() + ram_.ring_bytes();
}

uint64_t PagedEngine::get_largest_ram() const {
    const strata::core::PagePool& pool = ram_.pages().pool();
    const uint64_t chunk = pool.chunk_bytes();
    return (pool.report().total.bytes + get_ram_room(0)) / chunk * chunk + ram_.ring_bytes();
}

strata::spec::StepModel PagedEngine::get_step_model() {
    strata::spec::StepModel model;
    model.begin = [this](int count) { model_.begin_step(count); };
    model.make_room = [this](int session, int64_t tokens) {
        std::string err;
        if (time_step_if(settings_.step_profile != 0, pages_ms_,
                         [&] { return make_room(*held_[(size_t) session], tokens, err); }))
            return true;
        say("%s\n", err.c_str());
        return false;
    };
    model.run = [this](const std::vector<strata::spec::StepRow>& rows, std::vector<int32_t>& picks,
                       std::string& err) {
        step_rows_ = (int) rows.size();
        std::vector<const std::vector<int32_t>*> histories;
        histories.reserve(rows.size());
        for (const auto& row : rows)
            histories.push_back(&held_[(size_t) row.session]->live);
        model_.set_step_histories(histories);
        return time_step_if(settings_.step_profile != 0, verify_ms_,
                            [&] { return model_.run_rows(rows, picks, err); });
    };
    if (model_.get_pipeline_stages() > 1) {
        model.run_groups = [this](std::vector<strata::spec::StepGroup>& groups, std::string& err) {
            strata::spec::StageModel stages;
            stages.stages = model_.get_pipeline_stages();
            stages.prepare = [this](int group, strata::spec::StepGroup& g, std::string& err) {
                std::vector<const std::vector<int32_t>*> histories;
                for (const auto& row : g.rows)
                    histories.push_back(&held_[(size_t) row.session]->live);
                model_.set_step_histories(histories);
                return g.rows.empty() || model_.prepare_group(group, g.rows, err);
            };
            stages.launch = [this](int stage, int group, std::string& err) {
                return model_.launch_group(stage, group, err);
            };
            stages.poll = [this](int stage, int group, strata::spec::StepGroup& g, std::string& err) {
                return model_.poll_group(stage, group, g.picks, err);
            };
            stages.finish = [this](int group, strata::spec::StepGroup& g, std::string& err) {
                return model_.finish_group(group, g.picks, err);
            };
            step_rows_ = 0;
            for (const auto& g : groups)
                step_rows_ += (int) g.rows.size();
            // No admission, page trade, restore or settings command runs until this wave has drained.
            int peak = 0;
            stages.peak_in_flight = &peak;
            const bool ok = time_step_if(settings_.step_profile != 0, verify_ms_,
                [&] { return strata::spec::run_stage_groups(std::move(stages), groups, err); });
            if (peak > pipeline_peak_) {
                pipeline_peak_ = peak;
                say("pipeline: groups=%zu stages=%d peak_in_flight=%d\n", groups.size(),
                    model_.get_pipeline_stages(), peak);
            }
            return ok;
        };
        model.keep_group = [this](int group, const std::vector<strata::spec::StepRow>& rows,
                                  const std::vector<int>& kept, std::string& err) {
            return time_step_if(settings_.step_profile != 0, commit_ms_,
                [&] { return model_.keep_group(group, rows, kept, is_model_drafting(), err); });
        };
    }
    model.keep = [this](const std::vector<strata::spec::StepRow>& rows, const std::vector<int>& kept,
                        std::string& err) {
        return time_step_if(settings_.step_profile != 0, commit_ms_,
            [&] { return model_.keep_rows(rows, kept, is_model_drafting(), err); });
    };
    if (facts_.has_drafter)
        model.propose = [this](int session, int32_t token, int64_t pos, int n, int32_t* out,
                               strata::spec::DraftContext* draft) {
            if (!is_model_drafting())
                return 0;
            Held& h = *held_[(size_t) session];
            std::string err;
            if (n > 0 && !time_step_if(settings_.step_profile != 0, pages_ms_,
                                      [&] { return make_room(h, pos + 1 + n, err); }))
                n = 0;
            const int got = time_step_if(settings_.step_profile != 0, draft_ms_,
                [&] { return model_.propose(*h.parts.back(), session, token, pos, n, out, err, draft); });
            if (got >= 0)
                return got;
            say("the drafter: %s\n", err.c_str());
            return 0;
        };
    if (facts_.has_drafter && model_.has_batched_drafter())
        model.propose_batch = [this](std::vector<strata::spec::DraftRequest>& requests) {
            if (!is_model_drafting())
                return;
            std::string err;
            for (auto& request : requests) {
                Held& h = *held_[(size_t) request.session];
                if (request.limit > 0 && !time_step_if(settings_.step_profile != 0, pages_ms_,
                        [&] { return make_room(h, request.pos + 1 + request.limit, err); }))
                    request.limit = 0;
            }
            if (!time_step_if(settings_.step_profile != 0, draft_ms_,
                              [&] { return model_.propose_batch(requests, err); })) {
                for (auto& request : requests)
                    request.count = 0;
                say("the drafter: %s\n", err.c_str());
            }
        };
    return model;
}

std::optional<strata::spec::DraftPolicy> PagedEngine::find_draft_policy(
    const std::vector<int64_t>& ids, uint64_t image_key) const {
    /// Refers to a cached conversation without copying its token history.
    struct Candidate {
        const std::vector<int32_t>& live;
        const std::optional<strata::spec::DraftPolicy>& policy;
        uint64_t image_key, used;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(held_.size() + paused_.size());
    for (const auto& h : held_)
        candidates.push_back({h->live, h->policy, h->image_key, h->used});
    for (const Paused& p : paused_)
        candidates.push_back({p.head.live, p.policy, p.head.image_key, p.used});
    std::optional<strata::spec::DraftPolicy> policy;
    std::pair<size_t, uint64_t> best{};
    for (const Candidate& c : candidates) {
        const size_t n = std::min(c.live.size(), ids.size());
        if (c.policy && c.image_key == image_key && n > 0 && std::pair{n, c.used} > best &&
            std::equal(c.live.begin(), c.live.begin() + n, ids.begin())) {
            policy = c.policy;
            best = {n, c.used};
        }
    }
    return policy;
}

void PagedEngine::say(const char* format, ...) const {
    std::fprintf(stderr, "%s: ", facts_.name.c_str());
    va_list args;
    va_start(args, format);
    std::vfprintf(stderr, format, args);
    va_end(args);
}

/// Attributes synchronous caller time without adding clocks when profiling is disabled.
template <class F>
auto time_step_if(bool enabled, double& elapsed, F&& operation) -> decltype(operation()) {
    if (!enabled)
        return operation();
    const Clock::time_point start = Clock::now();
    auto result = operation();
    elapsed += get_ms_since(start);
    return result;
}

bool PagedEngine::wait_model_state() {
    std::string err;
    if (model_.wait_state(err) && fatal_error_.empty())
        return true;
    if (!fatal_error_.empty())
        err = fatal_error_;
    print_response("ERR %s\n", err.c_str());
    return false;
}

}  // namespace
}  // namespace strata::serve
