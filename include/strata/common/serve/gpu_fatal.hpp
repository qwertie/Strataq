// include/strata/common/serve/gpu_fatal.hpp - #4350: how an engine ends after a fatal GPU error.
//
// After an illegal memory access, a launch failure or a hardware exception the device context cannot be used again,
// and unwinding the destructors on it can hang until the stall watchdog (#224).  The engine therefore ends at once,
// with "ERR fatal GPU error: <reason>" as its last stdout line and exit code 3: the server fails the request in flight
// with a 503 and starts the engine again on the next request, like any engine death (strata_common/engine_client.py,
// `_raise_if_fatal_gpu`).  Used by strata (src/program/generate.cpp), strata-dsv41 (ds4_main.cpp) and the paged engine.
#pragma once

#include "strata/base/gpu/runtime.hpp"
#include "strata/common/serve/request_output.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace strata::serve {

/// Prints the fatal GPU error line for `reason` and ends the process with exit code 3, skipping destructors.
[[noreturn]] inline void exit_gpu_fatal(const std::string& reason) {
    std::fprintf(stderr, "strata serve: fatal GPU error (%s): the engine ends so that the server can start it again\n",
                 reason.c_str());
    std::fflush(stderr);
    print_response("ERR fatal GPU error: %s\n", reason.c_str());
    std::fflush(stdout);
    std::_Exit(3);
}

/// Ends the engine with exit_gpu_fatal when `err` (a failed step's message, or "" after one that succeeded) or the
/// runtime's sticky error says the GPU context is gone; returns otherwise.
/// STRATA_INJECT_FATAL=N (a test hook, off when unset): the Nth call ends the engine with the text of a real fatal
/// error; STRATA_INJECT_FATAL_ONCE=<file> makes that happen once across restarts (the file is made when it fires; while
/// it exists nothing is injected).
inline void exit_if_gpu_fatal(const std::string& err) {
    static const long inject = [] {
        const char* v = std::getenv("STRATA_INJECT_FATAL");
        return v != nullptr ? std::atol(v) : 0L;
    }();
    static long calls = 0;
    if (inject > 0 && ++calls == inject) {
        const char* once = std::getenv("STRATA_INJECT_FATAL_ONCE");
        bool fire = true;
        if (once != nullptr && *once) {
            if (std::FILE* t = std::fopen(once, "rb")) {
                std::fclose(t);
                fire = false;
            } else if (std::FILE* t2 = std::fopen(once, "wb")) {
                std::fclose(t2);
            }
        }
        if (fire)
            exit_gpu_fatal("injected (STRATA_INJECT_FATAL): an illegal memory access was encountered");
    }
    if (strata::core::is_context_lost_error(err))
        exit_gpu_fatal(err);
    if (const std::string sticky = strata::core::get_lost_context_error(); !sticky.empty())
        exit_gpu_fatal(err.empty() ? sticky : err);
}

/// Prints `err` as the ERR line of a failure, unless exit_if_gpu_fatal ends the engine for it first.
inline void print_error_or_exit(const std::string& err) {
    exit_if_gpu_fatal(err);
    print_response("ERR %s\n", err.c_str());
}

}  // namespace strata::serve
