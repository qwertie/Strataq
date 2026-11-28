// src/core/nv_defaults_test.cpp - the NVIDIA per-architecture defaults (no GPU needed): which cards get a table,
// and the K-quant prompt default and the prompt-chunk ceiling with their off switches.
#include "strata/base/platform/env.hpp"
#include "strata/core/nv_chunk.hpp"
#include "strata/core/nv_defaults.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

using strata::platform::set_env;
using strata::platform::unset_env;

static const std::string* find(const std::vector<std::pair<std::string, std::string>>& t, const char* k) {
    for (const auto& kv : t)
        if (kv.first == k) return &kv.second;
    return nullptr;
}

int main() {
    using namespace strata::core;
    unset_env("STRATA_SM86_PERF");
    unset_env("STRATA_SM60_PERF");
    unset_env("STRATA_EMULATE_CC");
    // only sm_86 and sm_60 have a table
    CHECK(nv_default_env(12, 1).empty());
    CHECK(nv_default_env(12, 0).empty());
    CHECK(nv_default_env(10, 0).empty());
    CHECK(nv_default_env(8, 9).empty());
    CHECK(nv_default_env(7, 5).empty());
    CHECK(!nv_default_env(8, 6).empty());
    CHECK(!nv_default_env(6, 0).empty());
    // round 0212: STRATA_MMQ_KQ=1 is a default for compute capability 8.0+ (owner yes 11-03), with its own off switches
    unset_env("STRATA_NV_KQ_DEFAULT");
    CHECK(kNvKqDefaultOn);
    for (int major : {8, 9, 10, 12}) CHECK(nv_kq_default_env(major, 0, false).size() == 1);   // on without any switch
    set_env("STRATA_NV_KQ_DEFAULT", "1");
    for (int major : {8, 9, 10, 12}) {
        const auto k = nv_kq_default_env(major, 0, false);
        CHECK(k.size() == 1 && find(k, "STRATA_MMQ_KQ") && *find(k, "STRATA_MMQ_KQ") == "1");
    }
    CHECK(nv_kq_default_env(7, 5, false).empty());     // Turing / Volta / Pascal keep the FP16 prompt path
    CHECK(nv_kq_default_env(6, 0, false).empty());
    CHECK(nv_kq_default_env(12, 0, true).empty());     // STRATA_DEFAULTS=026
    set_env("STRATA_NV_KQ_DEFAULT", "0");
    CHECK(nv_kq_default_env(12, 0, false).empty());    // its own off switch
    unset_env("STRATA_NV_KQ_DEFAULT");
    // the other tables are unchanged by it
    CHECK(nv_default_env(12, 0).empty());
    // round 0212: the NVIDIA prompt-chunk ceiling of a bare `--prefill auto` (nv_chunk.hpp)
    CHECK(nv_chunk_ceiling(nullptr, false) == 32768);
    CHECK(nv_chunk_ceiling("1", false) == 32768);
    CHECK(nv_chunk_ceiling("16", false) == 16384);
    CHECK(nv_chunk_ceiling("0", false) == 0);          // STRATA_NV_CHUNK=0 keeps 8192
    CHECK(nv_chunk_ceiling(nullptr, true) == 0);       // STRATA_DEFAULTS=026 keeps 8192
    CHECK(nv_chunk_ceiling("16", true) == 0);
    CHECK(!nv_chunk_card_ok(7) && !nv_chunk_card_ok(6) && nv_chunk_card_ok(8) && nv_chunk_card_ok(12));   // Volta / Turing / Pascal keep 8192
    CHECK(nv_chunk_step_down(32768) == 16384 && nv_chunk_step_down(16384) == 8192 && nv_chunk_step_down(8192) == 8192);
    if (fails == 0) std::printf("nv_defaults_test: ok\n");
    return fails == 0 ? 0 : 1;
}
