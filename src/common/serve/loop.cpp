// strata/common/serve/loop.cpp - see loop.hpp.
#include "strata/common/serve/loop.hpp"
#include "strata/common/serve/request_output.hpp"
#include "strata/common/serve/request.hpp"

#include "strata/base/platform/aux_cpus.hpp"
#include "strata/base/platform/thread_name.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(_WIN32)
#include <io.h>
#else
#include <cerrno>
#include <unistd.h>
#endif

namespace strata::serve {
namespace {
/// Gets the handler command inside a valid ASK envelope without copying a queued prompt.
std::string_view get_handler_command(std::string_view command) {
    if (command.starts_with("ASK ")) {
        const size_t end = command.find(' ', 4);
        if (end != std::string::npos && is_protocol_identity(std::string(command.substr(4, end - 4))))
            command.remove_prefix(end + 1);
    }
    return command;
}
}

struct Input::Shared {
    std::atomic<bool> stop{false};
    std::atomic<bool> stop_all{false};   ///< is_stop_requested(id) for every id
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    /// Counts submissions so completion cannot erase an already queued replay of the same identity.
    std::unordered_map<std::string, size_t> requests;
    std::unordered_set<std::string> cancelled;
    bool eof = false;
    std::vector<long long> stopped_ids;   ///< the ids of the `STOP <id>` lines not yet forgotten
    size_t seen = 0;   ///< the queue's length when take_slot_controls last looked (wait_for_line's "new line")
};

Input::Input(bool quit_ends_request) : shared_(std::make_shared<Shared>()) {
    std::thread([st = shared_, quit_ends_request] {
        platform::name_this_thread("strata-stdin");
        aux_cpus::pin_current_thread();
        // read(2) on the descriptor, not std::cin: glibc's exit() flushes every stdio stream and waits for
        // stdin's lock, which getline holds while it waits for input - an engine ending on an error (every
        // std::exit) would hang in exit() on Linux, and the server would wait for it forever
        std::string l, buf;
        char chunk[4096];
        auto getline_fd = [&](std::string& out) -> bool {
            for (;;) {
                const size_t nlpos = buf.find('\n');
                if (nlpos != std::string::npos) {
                    out.assign(buf, 0, nlpos);
                    buf.erase(0, nlpos + 1);
                    return true;
                }
#if defined(_WIN32)
                const int n = _read(0, chunk, (unsigned) sizeof chunk);
#else
                const ssize_t n = ::read(0, chunk, sizeof chunk);
                if (n < 0 && errno == EINTR) continue;
#endif
                if (n <= 0) {
                    if (buf.empty()) return false;
                    out.swap(buf);
                    buf.clear();
                    return true;
                }
                buf.append(chunk, (size_t) n);
            }
        };
        auto stop_every_request = [&] {
            st->stop.store(true);
            st->stop_all.store(true);
        };
        while (getline_fd(l)) {
            if (!l.empty() && l.back() == '\r') l.pop_back();
            if (l.rfind("CANCEL ", 0) == 0) {
                std::lock_guard<std::mutex> lk(st->mu);
                const std::string id = l.substr(7);
                if (st->requests.contains(id))
                    st->cancelled.insert(id);
                continue;
            }
            if (l == "STOP") {
                stop_every_request();
                continue;
            }
            if (quit_ends_request && l.rfind("STOP ", 0) == 0) {
                st->stop.store(true);
                std::lock_guard<std::mutex> lk(st->mu);
                st->stopped_ids.push_back(std::atoll(l.c_str() + 5));
                continue;
            }
            if (quit_ends_request && l == "QUIT")
                stop_every_request();
            std::lock_guard<std::mutex> lk(st->mu);
            RequestLine request;
            if (parse_request_line(l, request) && !request.request_id.empty())
                ++st->requests[request.request_id];
            st->lines.push_back(l);
            st->cv.notify_one();
        }
        std::lock_guard<std::mutex> lk(st->mu);
        st->eof = true;
        if (quit_ends_request)
            stop_every_request();
        st->cv.notify_one();
    }).detach();
}

bool Input::next(std::string& line) {
    std::unique_lock<std::mutex> lk(shared_->mu);
    shared_->cv.wait(lk, [&] { return !shared_->lines.empty() || shared_->eof; });
    if (shared_->lines.empty()) return false;
    line = std::move(shared_->lines.front());
    shared_->lines.pop_front();
    return true;
}

bool Input::try_next(std::string& line) {
    std::lock_guard<std::mutex> lk(shared_->mu);
    if (shared_->lines.empty()) return false;
    line = std::move(shared_->lines.front());
    shared_->lines.pop_front();
    return true;
}

std::vector<std::string> Input::take_inspections() {
    std::vector<std::string> lines;
    std::lock_guard<std::mutex> lock(shared_->mu);
    for (auto it = shared_->lines.begin(); it != shared_->lines.end();) {
        const std::string_view command = get_handler_command(*it);
        if (command == "GET_MEMORY_USE" || command.starts_with("LIST_CONVERSATIONS") ||
            command.starts_with("GET_CONTEXT_WINDOWS")) {
            lines.push_back(std::move(*it));
            it = shared_->lines.erase(it);
        } else
            ++it;
    }
    return lines;
}

bool Input::has_pending_settings() const {
    std::lock_guard<std::mutex> lock(shared_->mu);
    return std::any_of(shared_->lines.begin(), shared_->lines.end(), [](const std::string& line) {
        const auto command = get_handler_command(line);
        return command == "CHANGE_SETTINGS" || command.starts_with("CHANGE_SETTINGS ");
    });
}

bool Input::is_stop_requested() const {
    return shared_->stop.load() || is_cancel_requested(get_output_request());
}

bool Input::is_cancel_requested(const std::string& id) const {
    std::lock_guard<std::mutex> lk(shared_->mu);
    return shared_->cancelled.contains(id);
}

void Input::finish_request(const std::string& id) {
    std::lock_guard<std::mutex> lk(shared_->mu);
    const auto request = shared_->requests.find(id);
    if (request != shared_->requests.end() && --request->second == 0) {
        shared_->requests.erase(request);
        shared_->cancelled.erase(id);
    }
}

void Input::clear_stop() { shared_->stop.store(false); }

bool Input::is_stop_requested(long long request_id) const {
    if (shared_->stop_all.load()) return true;
    std::lock_guard<std::mutex> lk(shared_->mu);
    const auto& ids = shared_->stopped_ids;
    return std::find(ids.begin(), ids.end(), request_id) != ids.end();
}

void Input::forget_stop(long long request_id) {
    std::lock_guard<std::mutex> lk(shared_->mu);
    auto& ids = shared_->stopped_ids;
    ids.erase(std::remove(ids.begin(), ids.end(), request_id), ids.end());
}

void Input::take_slot_controls(const std::function<void(int)>& stop_slot, const std::function<bool(int)>& yield) {
    std::lock_guard<std::mutex> lk(shared_->mu);
    for (const auto& [place, id] : get_legacy_requests())
        if (shared_->cancelled.contains(id))
            stop_slot(place);
    auto& lines = shared_->lines;
    for (auto it = lines.begin(); it != lines.end();) {
        if (it->rfind("BSTOP ", 0) == 0) {
            stop_slot(std::atoi(it->c_str() + 6));
            it = lines.erase(it);
        } else if (it->rfind("BYIELD ", 0) == 0 && yield(std::atoi(it->c_str() + 7))) {
            it = lines.erase(it);
        } else {
            ++it;
        }
    }
    shared_->seen = lines.size();
}

void Input::wait_for_line(std::chrono::milliseconds timeout, const std::function<bool()>& done) {
    std::unique_lock<std::mutex> lk(shared_->mu);
    shared_->cv.wait_for(lk, timeout, [&] { return done() || shared_->lines.size() != shared_->seen; });
}

void Input::wake() {
    { std::lock_guard<std::mutex> lk(shared_->mu); }   // (a waiter between its check and its sleep sees the change)
    shared_->cv.notify_all();
}

bool Loop::next(std::string& line) {
    for (;;) {
        clear_output_request();
        for (const std::string& id : take_finished_legacy_requests())
            in_.finish_request(id);
        if (hooks_.stop_slot)
            in_.take_slot_controls(hooks_.stop_slot, [](int) { return false; });
        if (hooks_.is_busy && hooks_.is_busy()) {
            if (!in_.try_next(line)) {
                if (!hooks_.step()) return fail();
                continue;
            }
        } else if (!in_.next(line)) {
            return false;
        }
        if (!translate_protocol_command(line)) {
            clear_output_request();
            continue;
        }
        if (hooks_.before_line && !hooks_.before_line(line)) return fail();
        if (hooks_.stop_slot && line.rfind("BSTOP ", 0) == 0) {
            hooks_.stop_slot(std::atoi(line.c_str() + 6));
            continue;
        }
        if (hooks_.stop_slot && line.rfind("BYIELD", 0) == 0) {
            if (hooks_.on_late_yield) hooks_.on_late_yield();
            continue;
        }
        if (answer_inspection(line))
            continue;
        if (!hooks_.change_settings || !hooks_.describe_memory || line.rfind("CHANGE_SETTINGS", 0) != 0)
            return true;
        std::vector<std::pair<std::string, std::string>> values;
        std::string why, shortfall;
        if (!parse_settings_line(line, values, why) || !(why = hooks_.change_settings(values, shortfall)).empty()) {
            print_response("ERR %s\n", why.c_str());
            std::fflush(stdout);
            continue;
        }
        write_memory_report(shortfall);
    }
}

void Loop::poll_inspections() {
    for (std::string line : in_.take_inspections()) {
        const InspectionOutput output;
        if (translate_protocol_command(line) && !answer_inspection(line))
            print_response("ERR inspection is unavailable\n");
    }
}

bool Loop::answer_inspection(const std::string& line) {
    if (hooks_.list_conversations && core::is_conversation_command(line)) {
        core::answer_conversation_command(line, hooks_.list_conversations(),
            [](const std::string& reply) { print_response("%s", reply.c_str()); });
        return true;
    }
    if (line != "GET_MEMORY_USE" || !hooks_.describe_memory)
        return false;
    write_memory_report();
    return true;
}

void Loop::write_memory_report(const std::string& shortfall) {
    MemoryReport report = hooks_.describe_memory();
    report.shortfall = shortfall;
    const std::string text = format_memory_report(report,
        hooks_.list_conversations ? hooks_.list_conversations() : std::vector<core::HeldConversation>{});
    print_response("%s", text.c_str());
}

}  // namespace strata::serve
