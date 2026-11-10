// strata/common/serve/loop.hpp - the engine's side of the server's line protocol: the lines it reads on stdin.
//
// The server writes one command per line: requests (GEN, GENI, BGEN, BGENI: request.hpp), STOP (end the running
// request), BSTOP <slot> (end a batch slot), BYIELD <slot> (a prompt read gives way to a shorter one),
// LIST_CONVERSATIONS and GET_CONTEXT_WINDOWS (token_listing.hpp), SAVE / RESTORE, VRAM and QUIT.  `Input` reads them on
// a thread of its own, so that STOP, BSTOP and BYIELD reach a request that is still running; `Loop` hands an engine the
// commands that are its own and answers the others itself.  Each engine path (src/program/generate.cpp, src/models/qwen36/q36_engine.cu,
// src/models/deepseekv41/ds4_main.cpp) says what it does through `EngineHooks`.
#pragma once

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
    /// as usual) and the end of stdin also raise the stop flag, and `STOP <anything>` counts as STOP (and stops request
    /// <anything> alone: is_stop_requested(request_id)).
    explicit Input(bool quit_ends_request = false);

    /// Takes the oldest queued line, waiting for one; false once stdin has ended and the queue is empty.
    bool next(std::string& line);
    /// Takes the oldest queued line if there is one, without waiting.
    bool try_next(std::string& line);

    /// Tells whether STOP has come since the last clear_stop().
    bool is_stop_requested() const;
    /// Forgets an earlier STOP (one that came between requests is stale).
    void clear_stop();
    /// Tells whether request `request_id` of an engine that runs several at once must end: `STOP <request_id>` came
    /// (with `quit_ends_request`), or a bare STOP - or, with `quit_ends_request`, QUIT or the end of stdin - ended
    /// every request for good.
    bool is_stop_requested(long long request_id) const;
    /// Forgets the `STOP <request_id>` lines that came: the request is over, and its id may come again.
    void forget_stop(long long request_id);

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
    /// Tells whether work is running that goes on without a new line (batch slots decoding).  Empty: never.
    std::function<bool()> is_busy;
    /// Runs one step of that work (one decode window); false is a failure that ends the engine.
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
};

/// The engine's command loop: `while (loop.next(line))` gets the commands the engine handles itself (requests,
/// QUIT, ...) and meanwhile runs the busy work and answers BSTOP, BYIELD and the conversation commands.
class Loop {
public:
    /// Takes `in` and `hooks` by reference: both must outlive the loop.
    Loop(Input& in, const EngineHooks& hooks) : in_(in), hooks_(hooks) {}

    /// Gets the next command for the engine; false when stdin has ended or a hook failed (is_failed()).
    bool next(std::string& line);
    /// Tells whether next() returned false because a step or before_line failed.
    bool is_failed() const { return failed_; }

private:
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
