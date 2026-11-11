// include/strata/common/experts/card_rules.hpp - the rules for experts on several cards and in RAM, from the links'
// measured speeds (docs/MEMORY_MANAGER_PLAN.md, "Several GPUs" and "Experts leaving a memory", step M8): which card
// holds which experts, when a layer's experts held on another card are computed there in one hop instead of by the
// CPU, how fast a slow card is refilled, and whether a card's evicted expert is copied back to RAM.  No GPU.
#pragma once

#include "strata/base/gpu/links.hpp"
#include "strata/common/state/page_pool.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace strata::core {

/// How many times faster than reading it from the SSD again copying an evicted expert back from a card must be for
/// the copy to be made, when the decode-time adaptive tier (or anything but a prompt's buffers) evicts it: when the
/// two cost about the same, dropping wins, as the expert may not be needed for a while.
constexpr double kCopyBackFactor = 1.2;
/// The same for an expert a prompt's buffers evict (PagePool::lend): the refill after the prompt wants it back soon,
/// so a copy no slower than the SSD's read is worth making.
constexpr double kLendCopyBackFactor = 1.0;

/// Gets the milliseconds `bytes` take over a link that carries `gbps` (a huge time when it carries nothing).
double get_copy_ms(uint64_t bytes, double gbps);
/// Gets the milliseconds one hop to the card at the far end of `link` costs: `rows` rows of `in_bytes` there (the
/// layer's input), their outputs of `out_bytes` back, and the link's round trip.
double get_hop_ms(const LinkSpeed& link, int64_t rows, uint64_t in_bytes, uint64_t out_bytes);
/// Tells whether a layer's `experts` routed experts that another card holds are computed there in one hop of
/// `hop_ms` rather than by the CPU from RAM, at `cpu_ms` each.
bool is_hop_cheaper(double hop_ms, int experts, double cpu_ms);
/// Gets how many experts of `bytes` each a refill puts onto the card at the far end of `link` within `budget_ms` (at
/// least 1, at most `most`): a slow link's refill is paced by its bandwidth (a 2 MB expert takes about 0.6 ms over
/// PCIe 3.0 x4).  `most` when the link was not measured.
int get_refill_batch(const LinkSpeed& link, uint64_t bytes, double budget_ms, int most);
/// Tells whether an expert a card evicts is copied back to RAM, when copying it back measures `gain` times as fast as
/// reading it from the SSD (get_copy_back_gain): at least kLendCopyBackFactor times for an eviction a prompt's buffers
/// made (`lent`), else kCopyBackFactor times.
bool is_copy_back_worth(double gain, bool lent);
/// Gets how many times faster copying an expert from a card to RAM (`back_gbps`) is than reading it from the SSD
/// (`ssd_gbps`): huge when the SSD's speed is unknown (0), 0 when the copy's is.
double get_copy_back_gain(double back_gbps, double ssd_gbps);

/// Gets why a layer split whose cards run on `devices` (each card's GPU, in the cards' order) may not compute experts
/// in hops between its cards (--remote-experts on), or an empty string when it may.  Two cards on one GPU may not: a
/// hop runs beside a verify window that spins on the same GPU waiting for it, and on the Strix Halo's GPU such hops
/// stalled and Windows reset the GPU driver (a GPU timeout, LiveKernelEvent 141).
std::string get_hop_refusal(const std::vector<int>& devices);

/// One card's pool as the placement sees it.
struct CardPool {
    int64_t lb = 0, le = 0;   ///< the layers the card runs: [lb, le)
    uint64_t room = 0;        ///< the bytes its pool has for experts
};

/// Where the profile's experts live on the cards.
struct ExpertPlacement {
    /// Per expert (layer * n_expert + expert): the card that holds it, or -1 (RAM or the SSD)
    std::vector<int32_t> card;
    /// Per layer: the card that holds the layer's experts that its own card does not (its one hop), or -1
    std::vector<int32_t> remote;
    std::vector<uint64_t> used;   ///< per card: the bytes of experts placed on it
};

/// Places the experts of `ranking` (hottest first) on `cards`: each card takes its own layers' experts, hottest
/// first, as far as its room goes; then a card with room left takes the hottest experts left of another card's layers,
/// where `may_hold(owner, holder)` allows it (one hop from the layer's card to it is cheaper than the CPU computing one
/// expert), all of a layer's experts away from its own card on one card, so that the layer makes one hop.  Of the
/// cards allowed, a layer's first such expert goes to the one with the most room left.  `expert_bytes` per layer.
ExpertPlacement place_experts(const std::vector<CardPool>& cards, const std::vector<ExpertKey>& ranking,
                              int64_t n_layers, int64_t n_expert, const std::vector<uint64_t>& expert_bytes,
                              const std::function<bool(int owner, int holder)>& may_hold);

}  // namespace strata::core
