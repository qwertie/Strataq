// strata/common/serve/loop.cpp - see loop.hpp.
#include "strata/common/serve/loop.hpp"

#include "strata/base/platform/aux_cpus.hpp"
#include "strata/base/platform/thread_name.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <io.h>
#else
#include <cerrno>
#include <unistd.h>
#endif

namespace strata::serve {

struct Input::Shared {
    std::atomic<bool> stop{false};
    std::atomic<bool> stop_all{false};   ///< is_stop_requested(id) for every id
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
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

bool Input::is_stop_requested() const { return shared_->stop.load(); }

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
        if (hooks_.is_busy && hooks_.is_busy()) {
            if (!in_.try_next(line)) {
                if (!hooks_.step()) return fail();
                continue;
            }
        } else if (!in_.next(line)) {
            return false;
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
        if (hooks_.list_conversations && core::is_conversation_command(line)) {
            core::answer_conversation_command(line, hooks_.list_conversations());
            continue;
        }
        return true;
    }
}

}  // namespace strata::serve
