// src/common/experts/layer_split.cpp - the layer split's planner (layer_split.hpp).  The search, its tie rules and its
// retries are the old engine's `--layer-split auto` (src/models/qwen38/model.cpp), whose placements must not move.
#include "strata/common/experts/layer_split.hpp"

#include "strata/common/experts/split_rules.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::core {

SplitPlan plan_layer_split(const SplitProblem& p, const char* log_prefix) {
    namespace rules = strata::program::split_rules;
    const int ns = (int) p.cards.size();
    const int64_t L = p.n_layers;
    SplitPlan plan;
    if (ns < 2)
        return plan;
    // ---- THE COVERAGE CURVE: HOW MUCH ROUTED MASS A CACHE HOLDS.  The profile file carries a ranking without
    // counts, so each rank's mass is modelled: (r+1)^-1.2 (fitted on a 5080 + 3090), or with `coverage_curve` the
    // increment of H(f) = 1 - (1 - f)^b over the held fraction f, which a 4-way IQ3_S rig measured (14.9M decode
    // lookups: the top 5,805 pairs held 69.2% of the mass where the power law claimed 94.9%).
    const size_t n_ranked = p.ranked_layers.size();
    std::vector<double> mass(n_ranked);
    double total_mass = 0;
    for (size_t r = 0; r < n_ranked; ++r) {
        if (!p.coverage_curve) {
            total_mass += (mass[r] = std::pow((double) r + 1.0, -1.2));
            continue;
        }
        const double hi = std::pow(1.0 - (double) r / (double) n_ranked, p.cover_b);
        const double lo = std::pow(1.0 - (double) (r + 1) / (double) n_ranked, p.cover_b);
        total_mass += (mass[r] = hi - lo);
    }
    std::vector<int64_t> used((size_t) ns), held_cnt((size_t) ns, 0);
    // the predicted window time (ms) of a placement, and the routed mass its pools hold; leaves each card's fill in
    // `used` and `held_cnt`
    const auto predict = [&](const std::vector<int64_t>& at, double& held_mass, int64_t& held) -> double {
        std::vector<int64_t> capr((size_t) ns);
        for (int i = 0; i < ns; ++i) {
            const int64_t lb = i == 0 ? 0 : at[(size_t) i - 1];
            const int64_t le = i + 1 < ns ? at[(size_t) i] : L;
            capr[(size_t) i] = p.room(i, lb, le);
        }
        std::fill(used.begin(), used.end(), 0);
        std::fill(held_cnt.begin(), held_cnt.end(), 0);
        held_mass = 0;
        held = 0;
        std::vector<double> mass_i((size_t) ns, 0.0), held_i((size_t) ns, 0.0);
        std::vector<bool> full((size_t) ns, false);
        for (size_t r = 0; r < n_ranked; ++r) {
            const int64_t l = p.ranked_layers[r];
            int st = 0;
            while (st + 1 < ns && l >= at[(size_t) st]) ++st;
            mass_i[(size_t) st] += mass[r];
            if (full[(size_t) st]) continue;
            if (used[(size_t) st] + p.expert_bytes(l) > capr[(size_t) st]) {   // a card stops at its first misfit
                full[(size_t) st] = true;
                continue;
            }
            used[(size_t) st] += p.expert_bytes(l);
            held_mass += mass[r];
            held_i[(size_t) st] += mass[r];
            ++held;
            ++held_cnt[(size_t) st];
        }
        held_mass /= std::max(total_mass, 1e-9);
        double ms = 0.0;
        for (int i = 0; i < ns; ++i)
            ms += p.miss_ms * p.cards[(size_t) i].miss_scale *
                  std::max(0.0, mass_i[(size_t) i] - held_i[(size_t) i]) / std::max(total_mass, 1e-9);
        for (int i = 0; i < ns; ++i) {
            const int64_t lb = i == 0 ? 0 : at[(size_t) i - 1], le = i + 1 < ns ? at[(size_t) i] : L;
            ms += (double) (le - lb) * p.cards[(size_t) i].layer_ms;
        }
        return ms;
    };
    std::vector<int64_t> best, at((size_t) ns - 1);
    double best_ms = 1e30, best_mass = 0;
    int64_t best_held = 0, tried = 0;
    // TIES GO TO THE BALANCED PLACEMENT.  When every placement holds the whole profile the predicted window is the same
    // for all of them, and a prompt flows through the cards as a pipeline, so its speed is the slowest card's: on 4x
    // R9700 6,36,41 read a 32K prompt at 1,750 tok/s and 12,24,36 at 2,780, same decode.  Among placements whose time
    // agrees to 1e-6 ms the one with the smallest slowest card (layers x ms per layer) is kept.
    double best_max = 1e30;
    const auto stage_max = [&](const std::vector<int64_t>& cuts) {
        double m = 0;
        for (int i = 0; i < ns; ++i) {
            const int64_t lb = i == 0 ? 0 : cuts[(size_t) i - 1], le = i + 1 < ns ? cuts[(size_t) i] : L;
            m = std::max(m, (double) (le - lb) * p.cards[(size_t) i].layer_ms);
        }
        return m;
    };
    const double tie_eps = 1e-6;
    // each card's part of a candidate, from the fill `predict` just computed
    const auto parts_of = [&](const std::vector<int64_t>& cuts) {
        std::vector<SplitFill> parts((size_t) ns);
        for (int i = 0; i < ns; ++i) {
            parts[(size_t) i].lb = i == 0 ? 0 : cuts[(size_t) i - 1];
            parts[(size_t) i].le = i + 1 < ns ? cuts[(size_t) i] : L;
            parts[(size_t) i].used = used[(size_t) i];
            parts[(size_t) i].held = held_cnt[(size_t) i];
            parts[(size_t) i].room = p.room(i, parts[(size_t) i].lb, parts[(size_t) i].le);
        }
        return parts;
    };
    bool gate_on = (bool) p.can_start;   // off when no placement can start: then the fastest is kept
    int64_t min_stage_layers = 1;        // #1616: kMinStageLayers in the second pass, after a pick with a tiny card
    std::vector<int64_t> stage_floors;   // #2644: each card's share of the room (empty: off)
    // #1428: 2 and 3 cards - every placement priced, so the near tie can name a really different one
    std::vector<std::pair<double, std::vector<int64_t>>> priced;
    const auto consider = [&]() {
        if (min_stage_layers > 1 && rules::smallest_stage(at, L) < min_stage_layers)
            return;
        if (!stage_floors.empty() && rules::below_floors(at, L, stage_floors))
            return;
        double hm = 0;
        int64_t held = 0;
        const double ms = predict(at, hm, held);
        ++tried;
        if (ns <= 3) priced.emplace_back(ms, at);
        const double mx = stage_max(at);
        if (ms < best_ms - tie_eps || (ms <= best_ms + tie_eps && mx < best_max - tie_eps)) {
            if (gate_on && !p.can_start(parts_of(at))) return;   // #1094: not a placement the engine cannot start
            best = at; best_ms = std::min(ms, best_ms); best_mass = hm; best_held = held; best_max = mx;
        }
    };
    // the layers shared in proportion to speed: beyond four cards, and for a model too small for four stages
    const auto proportional = [&]() {
        double total = 0;
        for (const SplitCard& c : p.cards) total += 1.0 / c.layer_ms;
        double acc = 0;
        for (int i = 0; i + 1 < ns; ++i) {
            acc += 1.0 / p.cards[(size_t) i].layer_ms;
            at[(size_t) i] = std::clamp<int64_t>((int64_t) std::llround(acc / total * (double) L),
                                                 i == 0 ? 2 : at[(size_t) i - 1] + 1, L - (ns - 1 - i));
        }
        consider();
    };
    const auto search = [&]() {
        priced.clear();
        if (ns == 2) {
            for (int64_t k = 2; k < L; ++k) { at[0] = k; consider(); }
            if (best.empty()) proportional();   // a model of fewer than three layers enumerates nothing
        } else if (ns == 3) {
            for (int64_t k1 = 2; k1 + 1 < L; ++k1)
                for (int64_t k2 = k1 + 1; k2 < L; ++k2) { at[0] = k1; at[1] = k2; consider(); }
            if (best.empty()) proportional();
        } else if (ns == 4) {
            // All C(L-3,3) placements (15,180 at 48 layers).  A card's fill depends only on its own range and room, so
            // each range is walked once and memoised: 1,121 walks instead of 15,180 (14 ms against 740 ms).  This
            // branch prices the misses without the per-card link scale, as it always has.
            const size_t stride = (size_t) L + 1;
            std::vector<double> held_mass_at((size_t) ns * (size_t) L * stride, -1.0);
            std::vector<int64_t> held_at(held_mass_at.size(), 0);
            std::vector<int64_t> used_at(held_mass_at.size(), 0);
            const auto range_hold = [&](int i, int64_t lb, int64_t le) -> size_t {
                const size_t key = ((size_t) i * (size_t) L + (size_t) lb) * stride + (size_t) le;
                if (held_mass_at[key] < 0.0) {
                    const int64_t room = p.room(i, lb, le);
                    double hm = 0;
                    int64_t u = 0, cnt = 0;
                    for (size_t r = 0; r < n_ranked; ++r) {
                        const int64_t l = p.ranked_layers[r];
                        if (l < lb || l >= le) continue;
                        if (u + p.expert_bytes(l) > room) break;   // the fill stops where predict's does
                        u += p.expert_bytes(l);
                        hm += mass[r];
                        ++cnt;
                    }
                    held_mass_at[key] = hm;
                    held_at[key] = cnt;
                    used_at[key] = u;
                }
                return key;
            };
            const double denom = std::max(total_mass, 1e-9);
            for (int64_t k1 = 2; k1 + 2 < L; ++k1)
                for (int64_t k2 = k1 + 1; k2 + 1 < L; ++k2)
                    for (int64_t k3 = k2 + 1; k3 < L; ++k3) {
                        const size_t a = range_hold(0, 0, k1), b = range_hold(1, k1, k2);
                        const size_t c = range_hold(2, k2, k3), d = range_hold(3, k3, L);
                        const double hm = held_mass_at[a] + held_mass_at[b] + held_mass_at[c] + held_mass_at[d];
                        const double ms = p.miss_ms * (1.0 - hm / denom) + (double) k1 * p.cards[0].layer_ms +
                                          (double) (k2 - k1) * p.cards[1].layer_ms +
                                          (double) (k3 - k2) * p.cards[2].layer_ms +
                                          (double) (L - k3) * p.cards[3].layer_ms;
                        if (min_stage_layers > 1 && rules::smallest_stage({k1, k2, k3}, L) < min_stage_layers)
                            continue;
                        if (!stage_floors.empty() && rules::below_floors({k1, k2, k3}, L, stage_floors))
                            continue;
                        ++tried;
                        const double mx = stage_max({k1, k2, k3});
                        if (!(ms < best_ms - tie_eps || (ms <= best_ms + tie_eps && mx < best_max - tie_eps)))
                            continue;
                        if (gate_on) {
                            const int64_t lbs[4] = {0, k1, k2, k3};
                            const int64_t les[4] = {k1, k2, k3, L};
                            const size_t keys[4] = {a, b, c, d};
                            std::vector<SplitFill> parts((size_t) ns);
                            for (int i = 0; i < ns; ++i)
                                parts[(size_t) i] = {lbs[i], les[i], used_at[keys[i]], held_at[keys[i]],
                                                     p.room(i, lbs[i], les[i])};
                            if (!p.can_start(parts))
                                continue;
                        }
                        best = {k1, k2, k3};
                        best_ms = std::min(ms, best_ms);
                        best_max = mx;
                        best_mass = hm / denom;
                        best_held = held_at[a] + held_at[b] + held_at[c] + held_at[d];
                    }
            if (best.empty()) proportional();   // four cards into fewer than five layers
        } else {
            proportional();   // beyond four cards the placements outnumber any budget worth spending at start
        }
    };
    const auto reset_best = [&]() {
        best.clear();
        best_ms = 1e30;
        best_max = 1e30;
        best_mass = 0;
        best_held = 0;
    };
    search();
    // ---- #1616: a card with one or two layers holds a handful of experts, and the prompt chunk of every card is
    // clamped to what the smallest pool can lend: look again among placements with at least kMinStageLayers per card.
    if (!best.empty() && rules::restriction_possible(L, ns) && rules::has_tiny_stage(best, L)) {
        const std::vector<int64_t> first = best;
        const double first_ms = best_ms;
        // #2644 (two cards): first a placement in which each card owns at least its share of the layers by its room
        // for experts (4060 Ti 16 GB + 3060 12 GB: the three-layer rule's K=3 left the 3060 idle, -44% prefill); the
        // rooms are read at the even split
        bool share_pick = false;
        if (ns == 2) {
            std::vector<int64_t> rooms((size_t) ns);
            for (int i = 0; i < ns; ++i)
                rooms[(size_t) i] = p.room(i, L * i / ns, L * (i + 1) / ns);
            const std::vector<int64_t> floors = rules::share_floors(rooms, L);
            if (rules::floors_exceed_minimum(floors)) {
                reset_best();
                stage_floors = floors;
                search();
                stage_floors.clear();
                share_pick = !best.empty();
                if (share_pick)
                    std::fprintf(stderr, "%sK=%s would give a stage fewer than %lld layers (a tiny expert cache there "
                                         "clamps the prompt chunk of every card, #1616) and leave that card idle; "
                                         "chose K=%s instead, which gives each card at least its share of the layers "
                                         "by its room for experts (%.1f ms per decode window, not %.1f; --layer-split "
                                         "N sets it by hand, #2644)\n", log_prefix, format_cuts(first).c_str(),
                                 (long long) rules::kMinStageLayers, format_cuts(best).c_str(), best_ms, first_ms);
            }
        }
        if (!share_pick) {
            reset_best();
            min_stage_layers = rules::kMinStageLayers;
            search();
            min_stage_layers = 1;
            if (best.empty()) {
                best = first;
                best_ms = first_ms;
                std::fprintf(stderr, "%sK=%s gives a stage fewer than %lld layers (a tiny expert cache clamps the "
                                     "prompt chunk, #1616), but no placement with at least that many layers can "
                                     "start; keeping it\n", log_prefix, format_cuts(first).c_str(),
                             (long long) rules::kMinStageLayers);
            } else {
                std::fprintf(stderr, "%sK=%s would give a stage fewer than %lld layers (a tiny expert cache there "
                                     "clamps the prompt chunk of every card, #1616); chose K=%s instead (%.1f ms per "
                                     "decode window, not %.1f)\n", log_prefix, format_cuts(first).c_str(),
                             (long long) rules::kMinStageLayers, format_cuts(best).c_str(), best_ms, first_ms);
            }
        }
    }
    if (best.empty() && gate_on && p.choose_unchecked) {
        // recommend, never force: with nothing startable on paper the search runs again without the check
        std::fprintf(stderr, "%sno placement can start the prompt path on the figures here; choosing by decode speed "
                             "alone\n", log_prefix);
        gate_on = false;
        plan.start_unchecked = true;
        search();
    }
    if (best.empty())
        return plan;
    double hm = 0;
    int64_t held = 0;
    (void) predict(best, hm, held);   // the chosen placement's fill
    plan.fills = parts_of(best);
    plan.cuts = best;
    plan.ms = best_ms;
    plan.mass = best_mass;
    plan.held = best_held;
    plan.tried = tried;
    // #1428: a near tie on a mixed pair.  The model prices a decode window only, and on cards of different speed or
    // link the winner can flip within a few percent of error: name the other split that is almost as good.
    double lo = 1e30, hi = 0, llo = 1e30, lhi = 0;
    for (const SplitCard& c : p.cards) {
        lo = std::min(lo, c.layer_ms);
        hi = std::max(hi, c.layer_ms);
        llo = std::min(llo, c.miss_scale);
        lhi = std::max(lhi, c.miss_scale);
    }
    const bool mixed = hi > lo * 1.10 || lhi > llo * 1.10;
    if (mixed && p.near_tie_pct > 0 && ns <= 3 && best_ms > 0 && !priced.empty()) {
        double alt_ms = 1e30;
        for (const auto& pc : priced) {
            if (pc.first > best_ms * (1.0 + p.near_tie_pct / 100.0) || pc.first >= alt_ms) continue;
            int64_t dist = 0;   // how far the other split's cuts lie from the chosen one's
            for (size_t i = 0; i < pc.second.size() && i < best.size(); ++i)
                dist = std::max<int64_t>(dist, std::llabs(pc.second[i] - best[i]));
            if (dist < 3) continue;   // the neighbour one layer over is not a different split
            if (gate_on) {
                (void) predict(pc.second, hm, held);
                if (!p.can_start(parts_of(pc.second))) continue;
            }
            plan.near_tie = pc.second;
            plan.near_tie_ms = alt_ms = pc.first;
        }
    }
    return plan;
}

void print_layer_split(const SplitPlan& plan, size_t n_ranked, const char* log_prefix) {
    const std::string ks = format_cuts(plan.cuts);
    std::fprintf(stderr, "%sK=%s - predicted %.1f ms per decode window; the caches hold %lld of %zu profiled pairs "
                         "(~%.1f%% of the routed mass), best of %lld placements\n", log_prefix, ks.c_str(), plan.ms,
                 (long long) plan.held, n_ranked, 100.0 * plan.mass, (long long) plan.tried);
    if (plan.near_tie.empty())
        return;
    const std::string as = format_cuts(plan.near_tie);
    std::fprintf(stderr, "%sNEAR TIE - K=%s is predicted %.1f ms per decode window, within %.1f%% of K=%s (%.1f ms); "
                         "the cards differ, so the prediction may not rank them (it prices decode, not the prompt). To "
                         "try the other: --layer-split %s\n", log_prefix, as.c_str(), plan.near_tie_ms,
                 100.0 * (plan.near_tie_ms / plan.ms - 1.0), ks.c_str(), plan.ms, as.c_str());
}

std::string format_cuts(const std::vector<int64_t>& cuts) {
    std::string s;
    for (const int64_t k : cuts)
        s += (s.empty() ? "" : ",") + std::to_string(k);
    return s;
}

uint64_t LayerSizes::get_range_bytes(int64_t lb, int64_t le, int64_t conversations, int64_t tokens) const {
    uint64_t bytes = 0;
    for (int64_t l = lb; l < le; ++l)
        bytes += dense[(size_t) l] +
                 (uint64_t) conversations * (fixed_state[(size_t) l] + (uint64_t) tokens * token_bytes[(size_t) l]);
    return bytes;
}

SplitProblem make_paged_split_problem(const LayerSizes& sizes, const std::vector<uint64_t>& pool_bytes,
                                      int64_t conversations, int64_t tokens) {
    SplitProblem p;
    p.n_layers = (int64_t) sizes.expert.size();
    p.cards.resize(pool_bytes.size());
    p.expert_bytes = [expert = sizes.expert](int64_t layer) { return (int64_t) expert[(size_t) layer]; };
    p.room = [sizes, pool_bytes, conversations, tokens](int card, int64_t lb, int64_t le) {
        return (int64_t) pool_bytes[(size_t) card] - (int64_t) sizes.get_range_bytes(lb, le, conversations, tokens);
    };
    return p;
}

}  // namespace strata::core
