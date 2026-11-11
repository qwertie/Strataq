// The rules for experts on several cards (card_rules.cpp) on synthetic links: no GPU.  An RTX 3060 on PCIe 3.0 x4, an
// RTX 3080 on x16, a card in a Thunderbolt enclosure and a fast NVMe drive.
#include "strata/common/experts/card_rules.hpp"

#include <cmath>
#include <cstdio>

using namespace strata::core;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    const LinkSpeed x4{3.2, 3.0, 0.05}, x16{24.0, 22.0, 0.03}, thunderbolt{2.5, 2.3, 0.12};
    const double nvme_gbps = 3.0;
    const uint64_t expert = 2000000;   // a 2 MB expert
    // a 2 MB expert over PCIe 3.0 x4 takes about 0.6 ms
    CHECK(std::fabs(get_copy_ms(expert, x4.to_gbps) - 0.625) < 1e-9);
    CHECK(get_copy_ms(expert, 0) > 1e20);
    // refills: as many as a 10 ms budget carries, at least one, at most the batch
    CHECK(get_refill_batch(x16, expert, 10.0, 16) == 16);
    CHECK(get_refill_batch(x4, expert, 10.0, 16) == 16);
    CHECK(get_refill_batch(x4, expert, 5.0, 16) == 8);
    CHECK(get_refill_batch(thunderbolt, expert, 5.0, 16) == 6);
    CHECK(get_refill_batch(thunderbolt, expert, 0.1, 16) == 1);
    CHECK(get_refill_batch(LinkSpeed{}, expert, 5.0, 16) == 16);
    // copying an evicted expert back: worth it from an x16 card; from an x4 one, as fast as reading a fast NVMe drive
    // again, only when a prompt's buffers evicted it; never from a Thunderbolt enclosure, which is slower
    const double x16_gain = get_copy_back_gain(x16.from_gbps, nvme_gbps),
                 x4_gain = get_copy_back_gain(x4.from_gbps, nvme_gbps),
                 thunderbolt_gain = get_copy_back_gain(thunderbolt.from_gbps, nvme_gbps);
    CHECK(is_copy_back_worth(x16_gain, false) && is_copy_back_worth(x16_gain, true));
    CHECK(!is_copy_back_worth(x4_gain, false) && is_copy_back_worth(x4_gain, true));
    CHECK(!is_copy_back_worth(thunderbolt_gain, false) && !is_copy_back_worth(thunderbolt_gain, true));
    CHECK(is_copy_back_worth(get_copy_back_gain(thunderbolt.from_gbps, 0.5), false));   // a slow SATA drive
    CHECK(is_copy_back_worth(get_copy_back_gain(1.0, 0), false) && get_copy_back_gain(0, 3.0) == 0);
    CHECK(!is_copy_back_worth(1.199, false) && is_copy_back_worth(1.2, false));
    CHECK(!is_copy_back_worth(0.999, true) && is_copy_back_worth(1.0, true));
    // one hop of 8 rows (16 KiB in, 16 KiB out each) against the CPU at 0.1 ms per expert from RAM
    const uint64_t row = 16384;
    const double hop_x4 = get_hop_ms(x4, 8, row, row), hop_tb = get_hop_ms(thunderbolt, 8, row, row);
    CHECK(std::fabs(hop_x4 - (0.05 + 8 * row / 3.2e6 + 8 * row / 3.0e6)) < 1e-9);
    CHECK(!is_hop_cheaper(hop_x4, 1, 0.1) && is_hop_cheaper(hop_x4, 2, 0.1));
    CHECK(!is_hop_cheaper(hop_tb, 2, 0.1) && is_hop_cheaper(hop_tb, 4, 0.1));
    std::printf("hop of 8 rows: x4 %.3f ms, Thunderbolt %.3f ms\n", hop_x4, hop_tb);
    // hops between cards: refused only where two cards share a GPU (--split-device 0, or a GPU named twice)
    CHECK(get_hop_refusal({0, 1}).empty() && get_hop_refusal({0, 1, 2}).empty() && get_hop_refusal({0}).empty());
    CHECK(!get_hop_refusal({0, 0}).empty() && !get_hop_refusal({0, 1, 1}).empty());
    CHECK(get_hop_refusal({0, 2, 0}).find("cards 0 and 2 share GPU 0") == 0);
    {   // placement: layers 0-23 on card 0 (room for 3 experts), 24-47 on card 1 (room for 30)
        const int64_t layers = 48, n_expert = 4;
        const std::vector<uint64_t> bytes(layers, expert);
        std::vector<ExpertKey> ranking;
        for (int32_t e = 0; e < n_expert; ++e)
            for (int32_t l = 0; l < layers; ++l)
                ranking.push_back({l, e});
        const std::vector<CardPool> cards = {{0, 24, 3 * expert}, {24, 48, 30 * expert}};
        const auto at = [&](int l, int e) { return (size_t) (l * n_expert + e); };
        // no hop allowed: each card holds only its own layers' experts, hottest first
        ExpertPlacement p = place_experts(cards, ranking, layers, n_expert, bytes, [](int, int) { return false; });
        CHECK(p.used[0] == 3 * expert && p.used[1] == 30 * expert);
        CHECK(p.card[at(0, 0)] == 0 && p.card[at(2, 0)] == 0 && p.card[at(3, 0)] == -1);
        CHECK(p.card[at(24, 0)] == 1 && p.card[at(29, 1)] == 1 && p.card[at(30, 1)] == -1);
        for (const int32_t r : p.remote)
            CHECK(r == -1);
        // a hop from card 0 to card 1 allowed: card 1's room left takes card 0's hottest experts left, one card per
        // layer
        p = place_experts(cards, ranking, layers, n_expert, bytes, [](int owner, int holder) {
            return owner == 0 && holder == 1;
        });
        CHECK(p.used[1] == 30 * expert);
        CHECK(p.card[at(24, 0)] == 1 && p.card[at(29, 1)] == 1);   // its own first
        CHECK(p.card[at(3, 0)] == -1);                             // its room was gone by then
        // with room for every expert on card 1, layer 3's experts go there, and layer 3 hops to card 1
        const std::vector<CardPool> roomy = {{0, 24, 3 * expert}, {24, 48, 200 * expert}};
        p = place_experts(roomy, ranking, layers, n_expert, bytes, [](int owner, int holder) {
            return owner == 0 && holder == 1;
        });
        CHECK(p.card[at(3, 0)] == 1 && p.card[at(3, 3)] == 1 && p.remote[3] == 1 && p.remote[0] == 1);
        CHECK(p.remote[30] == -1);
        CHECK(p.used[0] + p.used[1] == layers * n_expert * expert);
    }
    if (failed == 0) std::printf("card_rules_test: all passed\n");
    return failed == 0 ? 0 : 1;
}
