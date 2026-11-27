// The route tail skip's rule (mark_tail_skips): no GPU, no model.
#include <cstdio>

#include "strata/common/experts/tail_skip.hpp"

using strata::core::mark_tail_skips;

static int failed = 0;
#define CHECK(x) \
    do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    // experts 0-9 on the GPU, 10 and up missing
    const auto missing = [](int32_t e) { return e >= 10; };
    bool skip[16];
    {   // one token ranked by order: rank 2 keeps the top 2, so missing 11 (rank 2) and 12 (rank 3) go, 13 stays
        const int32_t ids[4] = {13, 1, 11, 12};
        CHECK(mark_tail_skips(ids, nullptr, 1, 4, 4, 64, 2, missing, skip) == 2);
        CHECK(!skip[0] && !skip[1] && skip[2] && skip[3]);
        CHECK(mark_tail_skips(ids, nullptr, 1, 4, 4, 64, 0, missing, skip) == 0);   // off
        CHECK(!skip[0] && !skip[1] && !skip[2] && !skip[3]);
        CHECK(mark_tail_skips(ids, nullptr, 1, 4, 4, 64, 4, missing, skip) == 0);   // past the last rank
    }
    {   // a window: 11 ranks low for token 0 but high for token 1, so it stays for both; 12 is low for both
        const int32_t ids[6] = {1, 2, 11, 11, 12, 2};
        CHECK(mark_tail_skips(ids, nullptr, 2, 3, 3, 64, 1, missing, skip) == 1);
        CHECK(!skip[2] && !skip[3] && skip[4]);
    }
    {   // ranked by weight, rows 4 apart with 3 entries each: 11 ranks 2 for token 0 but 1 for token 1 (a tie with 13
        // that the earlier entry wins), so only 13 goes
        const int32_t ids[8] = {11, 12, 1, -1, 11, 13, 14, -1};
        const float w[8] = {0.1f, 0.5f, 0.4f, 0.0f, 0.2f, 0.2f, 0.6f, 0.0f};
        CHECK(mark_tail_skips(ids, w, 2, 3, 4, 64, 2, missing, skip) == 1);
        CHECK(!skip[0] && !skip[1] && !skip[2] && !skip[4] && skip[5] && !skip[6]);
    }
    {   // an id outside the model's experts is never marked
        const int32_t ids[2] = {1, 70};
        CHECK(mark_tail_skips(ids, nullptr, 1, 2, 2, 64, 1, missing, skip) == 0);
    }
    if (failed == 0)
        std::printf("tail_skip_test: ok\n");
    return failed == 0 ? 0 : 1;
}
