// strata/common/serve/loop.hpp - the engine's side of the server's line protocol: the lines it reads on stdin.
//
// The server sends RUN/RUNI identities, targeted CANCEL and ASK command envelopes (contracts/requests.md).
// Input reads independently of model work. Loop services commands between decode steps; older handler forms
// are translated by request_output.hpp for engines that configure its compatibility adapter.
#pragma once

#include "strata/common/serve/memory_report.hpp"
#include "strata/common/state/token_listing.hpp"

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::serve {

/// The command lines the server has sent and the engine has not taken yet, read from stdin on a detached thread.
/// A `STOP` line is not queued: it raises the stop flag, which a running request checks between its steps.
class Input {
public:
    /// Starts the reading thread.  It may outlive this object (it can sit in a read when the process ends), so what
    /// it shares with the engine is kept alive by both.  With `quit_ends_request` (DeepSeek's habit), QUIT (queued
    /// as usual) and the end of stdin also raise the stop flag, and `STOP <anything>` counts as STOP.
    explicit Input(bool quit_ends_request = false);

    /// Takes the oldest queued line, waiting for one; false once stdin has ended and the queue is empty.
    bool next(std::string& line);
    /// Takes the oldest queued line if there is one, without waiting.
    bool try_next(std::string& line);
    /// Removes queued host-only inspection commands without admitting generation or settings changes.
    std::vector<std::string> take_inspections();

    /// Tells whether STOP has come since the last clear_stop().
    bool is_stop_requested() const;
    /// Forgets an earlier STOP (one that came between requests is stale).
    void clear_stop();
    /// Checks targeted cancellation independently of which request currently uses the model.
    bool is_cancel_requested(const std::string& id) const;
    /// Releases cancellation bookkeeping after a request's terminal event.
    void finish_request(const std::string& id);

    /// Removes the queued `BSTOP <slot>` lines, giving each slot to `stop_slot`, and the queued `BYIELD <slot>` lines
    /// that `yield` accepts (it returns false to leave a line queued).  Used by a prompt read that batch slots
    /// decode beside, which must see these lines before the read ends.
    void take_slot_controls(const std::function<void(int slot)>& stop_slot, const std::function<bool(int slot)>& yield);
    /// Waits until a line comes that take_slot_controls has not looked at, `done()` holds or `timeout` passes.
    /// Whoever makes `done()` true calls wake().
    void wait_for_line(std::chrono::milliseconds timeout, const std::function<bool()>& done);
    /// Wakes wait_for_line() to check its `done` again.
    void wake();

private:
    struct Shared;
    std::shared_ptr<Shared> shared_;
};

/// What an engine tells the shared Loop: the work that goes on between lines, and the answers to the commands the
/// loop handles itself.  An empty function is a feature the engine does not have.
struct EngineHooks {
    /// Tells whether work is running that goes on without a new line (batch slots decoding, strata-q38's experts
    /// going back to the GPU).  Empty: never.
    std::function<bool()> is_busy;
    /// Runs one step of that work (one decode window, one batch of experts); false is a failure that ends the engine.
    std::function<bool()> step;
    /// Sees every line before the loop looks at it; false is a failure that ends the engine.
    std::function<bool(const std::string& line)> before_line;
    /// Ends batch slot `slot` (not range-checked) at its next window: `BSTOP <slot>`.  Empty: the engine has no batch
    /// slots, and BSTOP and BYIELD lines go to it as commands.
    std::function<void(int slot)> stop_slot;
    /// Sees a BYIELD that reached the loop, which drops it: the prompt read it was meant for has ended.
    std::function<void()> on_late_yield;
    /// Describes the conversations the engine holds, for LIST_CONVERSATIONS and GET_CONTEXT_WINDOWS.
    std::function<std::vector<core::HeldConversation>()> list_conversations;
    /// Describes the engine's memories and settings for GET_MEMORY_USE (memory_report.hpp); the parked conversations
    /// of list_conversations are added to it.
    std::function<MemoryReport()> describe_memory;
    /// Changes the settings `values` names (CHANGE_SETTINGS' fields, in order) all at once; gets why not (having
    /// changed none of them), or an empty string.  `shortfall` gets what an accepted change got less of than asked
    /// (MemoryReport::shortfall).  The loop then answers with describe_memory's report.  Empty: the engine changes
    /// nothing while it runs, and CHANGE_SETTINGS lines go to it as commands.
    std::function<std::string(const std::vector<std::pair<std::string, std::string>>& values, std::string& shortfall)>
        change_settings;
};

/// The engine's command loop: `while (loop.next(line))` gets the commands the engine handles itself (requests,
/// QUIT, ...) and meanwhile runs the busy work and answers BSTOP, BYIELD and the conversation commands.
class Loop {
public:
    /// Takes `in` and `hooks` by reference: both must outlive the loop.
    Loop(Input& in, const EngineHooks& hooks) : in_(in), hooks_(hooks) {}

    /// Gets the next command for the engine; false when stdin has ended or a hook failed (is_failed()).
    bool next(std::string& line);
    /// Answers host-only inspection at a model's safe prompt-chunk boundary.
    void poll_inspections();
    /// Tells whether next() returned false because a step or before_line failed.
    bool is_failed() const { return failed_; }

private:
    /// Answers a supported read-only command; returns false for other commands.
    bool answer_inspection(const std::string& line);
    /// Sends the current allocation report, including any partial settings-change result.
    void write_memory_report(const std::string& shortfall = {});

    /// Records a failed hook; returns next()'s false.
    bool fail() {
        failed_ = true;
        return false;
    }

    Input& in_;
    const EngineHooks& hooks_;
    bool failed_ = false;
};

}  // namespace strata::serve
