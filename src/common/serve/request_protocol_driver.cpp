// Exercises the real command loop and transport without a model or GPU.
#include "strata/common/serve/loop.hpp"
#include "strata/common/serve/request.hpp"
#include "strata/common/serve/request_output.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <set>
#include <thread>
#include <vector>

using namespace strata::serve;

namespace {
/// Retains just enough execution state to exercise interleaved request events.
struct Request {
    std::string id;
    int place = 0;
    int produced = 0;
    int maximum = 0;
    int token = 0;
    bool cancelled = false;
};
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 && std::string(argv[argc - 1]) != "--serve" ?
                             argv[argc - 1] : "native";
    const bool legacy = mode == "legacy", serial = mode == "serial";
    const bool prefill = mode == "prefill" || mode == "settings-prefill";
    if (legacy || serial)
        configure_legacy_requests(legacy ? 2 : 0);
    std::printf("INFO request_capacity=%d concurrent=2 context_capacity=8192\n", serial ? 1 : 2);
    if (!legacy && !serial)
        std::printf("INFO request_queue=1\n");
    std::printf("READY 8192 stop memory settings requests%s\n", prefill ? " conversations" : "");
    std::fflush(stdout);
    Input input;
    std::vector<Request> running;
    int rows = 2;
    std::set<std::string> preempted;
    std::vector<int32_t> context{31, 32};
    EngineHooks hooks;
    if (prefill)
        hooks.list_conversations = [&] {
            return std::vector<strata::core::HeldConversation>{
                {"conversation0", "vram", "reading", context.data(), context.size(), 8192, 1024, 1024}};
        };
    hooks.is_busy = [&] { return !running.empty(); };
    hooks.step = [&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        for (Request& request : running) {
            request.cancelled |= input.is_cancel_requested(request.id);
            const RequestOutput output(legacy ? "" : request.id);
            if (!request.cancelled) {
                if (legacy)
                    print_response("BT %d %d\n", request.place, request.token);
                else
                    print_response("T %d\n", request.token);
                ++request.produced;
            }
            const bool preempt = mode == "preempt" && request.produced == 3 &&
                                 preempted.insert(request.id).second;
            if (preempt || request.cancelled || request.produced == request.maximum) {
                input.finish_request(request.id);
                if (legacy)
                    print_response("BDONE %d %d %s 1\n", request.place, request.produced,
                                   request.cancelled ? "cancel" : "length");
                else if (mode == "settings-prefill")
                    print_response("DONE %d 2 1 1 %s 0 0 0 0 0 0 0 0.0 1 0\n", request.produced,
                                   request.cancelled ? "cancel" : "length");
                else
                    print_response("DONE %d 2 1 1 %s\n", request.produced,
                                   preempt ? "preempted" : request.cancelled ? "cancel" : "length");
                request.maximum = 0;
            }
        }
        std::erase_if(running, [](const Request& request) { return request.maximum == 0; });
        return true;
    };
    if (legacy)
        hooks.stop_slot = [&](int place) {
            for (Request& request : running)
                if (request.place == place)
                    request.cancelled = true;
        };
    hooks.describe_memory = [&] {
        if (mode == "slow" && rows == 2)
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
        MemoryReport report;
        report.settings = {{"rows", rows}, {"tokens_max", 8192}};
        return report;
    };
    hooks.change_settings = [&](const auto& changes, std::string&) {
        if (changes.size() != 1 || changes.front().first != "rows")
            return std::string("expected rows");
        rows = std::stoi(changes.front().second);
        return std::string{};
    };
    Loop loop(input, hooks);
    std::string line;
    while (loop.next(line)) {
        if (line == "QUIT")
            break;
        RequestLine parsed;
        if (!parse_request_line(line, parsed)) {
            print_response("ERR expected RUN\n");
            continue;
        }
        const std::string id = legacy || serial ? get_output_request() : parsed.request_id;
        const RequestOutput output(id);
        if (parsed.max_new < 1) {
            print_response("ERR max_new must be positive\n");
            input.finish_request(id);
            continue;
        }
        if (!legacy && !serial)
            print_response("STARTED\n");
        const int token = std::stoi(parsed.ids);
        print_response("PP 2 2 1 2000\n");
        bool yielded = false;
        if (prefill)
            for (int chunk = 0; chunk < 100; ++chunk) {
                std::this_thread::sleep_for(std::chrono::milliseconds(chunk == 0 ? 150 : 10));
                loop.poll_inspections();
                if (mode == "settings-prefill" && input.has_pending_settings()) {
                    input.finish_request(id);
                    print_response("DONE 0 2 1 0 preempted 0 0 0 0 0 0 0 0.0 1 0\n");
                    yielded = true;
                    break;
                }
                print_response("PP 2 2 1 2000\n");
            }
        if (yielded)
            continue;
        if (serial) {
            int produced = 0;
            while (produced < parsed.max_new && !input.is_stop_requested()) {
                print_response("T %d\n", token);
                ++produced;
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            print_response("DONE %d 2 1 1 %s\n", produced, input.is_stop_requested() ? "cancel" : "length");
            input.finish_request(id);
        } else {
            running.push_back({id, (int) parsed.slot, 0, (int) parsed.max_new, token, false});
            if (legacy) {
                print_response("DONE 0 2 1 0 length\n");
                print_response("BADM %ld 1\n", parsed.slot);
            }
        }
    }
    return loop.is_failed() ? 1 : 0;
}
