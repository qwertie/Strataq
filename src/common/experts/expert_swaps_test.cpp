// The adaptive tier's choice of exchanges (expert_swaps.hpp): the most-routed experts without a slot take the slots
// of the least-routed ones of their layer, only by a clear margin, largest gain first.
#include "strata/common/experts/expert_swaps.hpp"

#include <cstdio>
#include <cstdlib>

using namespace strata::core;

namespace {
int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) {
        std::fprintf(stderr, "FAIL: %s\n", description);
        std::exit(1);
    }
}

bool is_swap(const ExpertSwap& s, int32_t layer, int32_t in, int32_t out) {
    return s.layer == layer && s.in == in && s.out == out;
}
}  // namespace

int main() {
    // 3 layers of 4 experts; per layer, experts 0 and 1 have slots
    const std::vector<int32_t> res = {0, 1, -1, -1, 2, 3, -1, -1, 4, 5, -1, -1};
    const std::vector<float> usage = {0.0f, 4.0f, 5.0f, 1.9f,     // layer 0: 2 beats 0 by 5; 3 is routed too seldom
                                      3.0f, 9.0f, 4.0f, 0.0f,     // layer 1: 2 beats 0 by only 1
                                      1.0f, 2.0f, 9.0f, 8.0f};    // layer 2: 2 and 3 beat 0 and 1 by 8 and 6
    const auto none = [](int64_t, int32_t) { return false; };
    const std::vector<ExpertSwap> swaps = choose_expert_swaps(usage, res, 3, 4, 1.5f, none);
    check(swaps.size() == 3 && is_swap(swaps[0], 2, 2, 0) && is_swap(swaps[1], 2, 3, 1) && is_swap(swaps[2], 0, 2, 0),
          "the most-routed take the least-routed experts' slots, largest gain first; an even pair keeps its slot");
    check(swaps[0].gain == 8.0f && swaps[2].gain == 5.0f, "the gain is the difference of the routing counts");
    const std::vector<ExpertSwap> kept = choose_expert_swaps(usage, res, 3, 4, 1.5f,
                                                             [](int64_t l, int32_t e) { return l == 2 && e == 2; });
    check(kept.size() == 2 && is_swap(kept[0], 2, 3, 0) && is_swap(kept[1], 0, 2, 0),
          "an expert held elsewhere is no candidate");
    const std::vector<ExpertSwap> free = choose_expert_swaps(usage, res, 3, 4, 1.5f, none, true);
    check(free.size() == 4 && is_swap(free[2], 0, 2, 0) && is_swap(free[3], 1, 2, -1) && free[3].gain == 4.0f,
          "unpaired: a candidate without a partner it beats by the margin comes for free pages, its gain its count");
    check(choose_expert_swaps(std::vector<float>{3.0f, 0.0f}, std::vector<int32_t>{-1, -1}, 1, 2, 1.5f, none, true)
                  .size() == 1,
          "unpaired: a layer with no expert on the GPU still brings in its candidates");
    std::printf("expert_swaps_test: %d checks passed\n", checks);
    return 0;
}
