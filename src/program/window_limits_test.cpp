// The window limits (include/strata/common/kernels/window_limits.hpp): CPU only, header-only.
#include "strata/common/kernels/cpu/expert.hpp"
#include "strata/models/qwen38/kernels/fused_gr.hpp"
#include "strata/models/qwen38/kernels/verify_kernels.hpp"

#include <cstdio>
#include <cstdlib>

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
}  // namespace

int main() {
    using namespace strata::kernels;
    check(kVerifyNarrowT == 8, "the narrow window is the 0.2.7 one: 8 rows");
    check(kWindowMaxT == 8 || kWindowMaxT == 12 || kWindowMaxT == 16, "a window holds 8, 12 or 16 rows");
    check(kVerifyMaxT == kWindowMaxT, "kVerifyMaxT is the window limit");
    check(kVerifyMaxT >= kVerifyNarrowT, "the wide window is not narrower than the narrow one");
    check(cpu::MAXT == 8, "the CPU kernels' tokens per job stay 8 (a wider window is split into jobs)");
    check(kFusedGrMaxT == 8, "the hyper-connection kernels are built for 8 tokens (a wider window is read in slices)");
    check(kFusedGrWideT == kWindowMaxT, "the caller's token arrays follow the window");
    check(kVerifyMaxT * 10 <= 256, "the resident plan's block holds a full window of 10 routed experts a row");

    std::printf("window_limits_test: %d checks passed (window %d rows)\n", checks, (int) kVerifyMaxT);
    return 0;
}
