// src/common/experts/card_rules.cpp - the rules for experts on several cards and in RAM (card_rules.hpp).
#include "strata/common/experts/card_rules.hpp"

#include <algorithm>
#include <cmath>

namespace strata::core {

double get_copy_ms(uint64_t bytes, double gbps) { return gbps > 0 ? (double) bytes / (gbps * 1e6) : 1e30; }

double get_hop_ms(const LinkSpeed& link, int64_t rows, uint64_t in_bytes, uint64_t out_bytes) {
    return link.latency_ms + get_copy_ms((uint64_t) rows * in_bytes, link.to_gbps) +
           get_copy_ms((uint64_t) rows * out_bytes, link.from_gbps);
}

bool is_hop_cheaper(double hop_ms, int experts, double cpu_ms) { return hop_ms < experts * cpu_ms; }

int get_refill_batch(const LinkSpeed& link, uint64_t bytes, double budget_ms, int most) {
    if (link.to_gbps <= 0)
        return most;
    return std::clamp((int) (budget_ms / get_copy_ms(bytes, link.to_gbps)), 1, most);
}

bool is_copy_back_worth(double gain, bool lent) { return gain >= (lent ? kLendCopyBackFactor : kCopyBackFactor); }

double get_copy_back_gain(double back_gbps, double ssd_gbps) {
    return back_gbps <= 0 ? 0.0 : ssd_gbps > 0 ? back_gbps / ssd_gbps : HUGE_VAL;
}

std::string get_hop_refusal(const std::vector<int>& devices) {
    for (size_t i = 0; i < devices.size(); ++i)
        for (size_t j = i + 1; j < devices.size(); ++j)
            if (devices[i] == devices[j])
                return "cards " + std::to_string(i) + " and " + std::to_string(j) + " share GPU " +
                       std::to_string(devices[i]) + ": a hop between them waits behind a verify window spinning on " +
                       "that GPU (on the Strix Halo the hops stalled and Windows reset the GPU driver)";
    return {};
}

ExpertPlacement place_experts(const std::vector<CardPool>& cards, const std::vector<ExpertKey>& ranking,
                              int64_t n_layers, int64_t n_expert, const std::vector<uint64_t>& expert_bytes,
                              const std::function<bool(int owner, int holder)>& may_hold) {
    ExpertPlacement p;
    p.card.assign((size_t) (n_layers * n_expert), -1);
    p.remote.assign((size_t) n_layers, -1);
    p.used.assign(cards.size(), 0);
    std::vector<int> owner((size_t) n_layers, -1);
    for (size_t c = 0; c < cards.size(); ++c)
        for (int64_t l = cards[c].lb; l < cards[c].le && l < n_layers; ++l)
            owner[(size_t) l] = (int) c;
    const auto place = [&](const ExpertKey& k, int c) {
        const uint64_t bytes = expert_bytes[(size_t) k.layer];
        if (c < 0 || p.used[(size_t) c] + bytes > cards[(size_t) c].room)
            return false;
        p.used[(size_t) c] += bytes;
        p.card[(size_t) (k.layer * n_expert + k.expert)] = c;
        return true;
    };
    for (const ExpertKey& k : ranking)   // each card its own layers' experts
        (void) place(k, owner[(size_t) k.layer]);
    for (const ExpertKey& k : ranking) {   // then the room left, one other card per layer
        const int o = owner[(size_t) k.layer];
        if (o < 0 || p.card[(size_t) (k.layer * n_expert + k.expert)] >= 0)
            continue;
        int32_t& r = p.remote[(size_t) k.layer];
        if (r < 0) {
            uint64_t most = 0;
            for (int c = 0; c < (int) cards.size(); ++c) {
                const uint64_t left = cards[(size_t) c].room - p.used[(size_t) c];
                if (c != o && left >= expert_bytes[(size_t) k.layer] && left > most && may_hold(o, c)) {
                    most = left;
                    r = c;
                }
            }
        }
        (void) place(k, r);
    }
    return p;
}

}  // namespace strata::core
