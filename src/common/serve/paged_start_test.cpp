// Checks paged option choices and prompt-order environment precedence without starting a model or GPU runtime.
#include "strata/common/serve/paged_engine.hpp"
#include "strata/base/platform/env.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {
/// Stops the test at the first failed condition.
void check(bool ok, const char* description);
/// Parses one option with a fresh settings object and verifies that unowned arguments survive.
int parse_option(const char* flag, const char* value);
}  // namespace

int main() {
    for (const char* flag : {"--copy-back", "--remote-experts", "--prompt-order", "--drafter",
                             "--save-on-shutdown"}) {
        for (const char* value : {"auto", "on", "off", "layer", "chunk", "", "AUTO", "a", "on|off",
                                  "auto|on", "auto|layer", "|on", "off|"}) {
            const std::string f = flag, v = value;
            const bool valid = f == "--prompt-order" ? (v == "auto" || v == "layer" || v == "chunk") :
                (v == "on" || v == "off" ||
                 (v == "auto" && (f == "--copy-back" || f == "--remote-experts")));
            check(parse_option(flag, value) == (valid ? -1 : 2), "word-choice acceptance");
        }
    }
    for (const char* value : {"exact", "speculative", "typical", "lower-temperature", "top-75", "top-0", "top-100",
                              "on", ""}) {
        const std::string v = value;
        const bool valid = v != "top-0" && v != "top-100" && v != "on";   // (empty: the model's rule)
        check(parse_option("--draft-rule", value) == (valid ? -1 : 2), "draft-rule acceptance");
    }
    const auto label = strata::serve::get_expert_labels(40);
    check(label({39, 7}, 1 << 20).category == "experts", "trunk experts' category");
    check(label({40, 7}, 1 << 20).category == "drafter_experts", "drafter experts' category");
    const char* const names[] = {"STRATA_PREFILL_LAYER_MAJOR", "STRATA_LM_HIP", "DS4_LAYER_MAJOR"};
    for (const char* order : {"auto", "layer", "chunk"}) {
        strata::serve::PagedSettings settings;
        char program[] = "paged_start_test", flag[] = "--prompt-order";
        std::string value = order;
        char* argv[] = {program, flag, value.data(), nullptr};
        int argc = 3;
        check(strata::serve::take_paged_options(argc, argv, program, settings) == -1, "parse prompt order");
        check(settings.prompt_order == order, "store prompt order");
        for (const char* name : names)
            strata::platform::unset_env(name);
        strata::serve::set_paged_runtime_env(settings, false);
        for (const char* name : names) {
            const char* actual = std::getenv(name);
            check(settings.prompt_order == "auto" ? actual == nullptr :
                  actual != nullptr && std::string(actual) == (settings.prompt_order == "layer" ? "1" : "0"),
                  "map prompt order");
        }
        for (const char* explicit_value : {"0", "1", "-2"}) {
            for (const char* name : names)
                strata::platform::set_env(name, explicit_value);
            strata::serve::set_paged_runtime_env(settings, false);
            for (const char* name : names)
                check(std::string(std::getenv(name)) == explicit_value, "keep explicit environment");
        }
    }
    std::puts("paged_start_test: passed");
    return 0;
}

namespace {
void check(bool ok, const char* description) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", description);
        std::exit(1);
    }
}

int parse_option(const char* flag, const char* value) {
    std::string program = "paged_start_test", option = flag, word = value, unowned = "--model-option";
    char* argv[] = {program.data(), option.data(), word.data(), unowned.data(), nullptr};
    int argc = 4;
    strata::serve::PagedSettings settings;
    const int status = strata::serve::take_paged_options(argc, argv, program, settings);
    if (status == -1)
        check(argc == 2 && std::string(argv[1]) == unowned, "preserve unowned arguments");
    return status;
}
}  // namespace
