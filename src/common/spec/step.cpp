// src/common/spec/step.cpp - the decode step for several conversations at once (step.hpp).
#include "strata/common/spec/step.hpp"

#include <algorithm>
#include <chrono>

namespace strata::spec {

std::vector<int> share_guess_rows(const StepSettings& s, int active, int ties, int turn) {
    if (active <= 0)
        return {};
    ties = std::clamp(ties, 1, active);
    const int left = std::max(s.rows - active, 0);
    std::vector<int> share((size_t) active);
    for (int i = 0; i < active; ++i) {
        const int rank = i < ties ? ((i - turn) % ties + ties) % ties : i;   // in the order the extra rows go
        share[(size_t) i] = std::min(left / active + (rank < left % active ? 1 : 0), s.lookahead);
    }
    return share;
}

bool run_stage_groups(StageModel model, std::vector<StepGroup>& groups, std::string& err) {
    if (model.stages < 1) {
        err = "pipeline needs at least one stage";
        return false;
    }
    for (size_t g = 0; g < groups.size(); ++g)
        if (!model.prepare((int) g, groups[g], err))
            return false;
    std::vector<int> owner((size_t) model.stages, -1), next(groups.size(), 0);
    int left = (int) groups.size();
    for (size_t g = 0; g < groups.size(); ++g)
        if (groups[g].rows.empty()) {
            next[g] = model.stages;
            --left;
        }
    bool ok = true;
    while (left > 0 || std::any_of(owner.begin(), owner.end(), [](int g) { return g >= 0; })) {
        for (int stage = model.stages; stage-- > 0;) {
            int& g = owner[(size_t) stage];
            if (g < 0)
                continue;
            std::string problem;
            const int done = model.poll(stage, g, groups[(size_t) g], problem);
            if (done == 0)
                continue;
            if (done < 0) {
                if (ok)
                    err = problem;
                ok = false;
            }
            next[(size_t) g] = stage + 1;
            if (stage + 1 == model.stages)
                --left;
            g = -1;
        }
        if (!ok) {
            left = 0;
            continue;
        }
        for (int stage = model.stages; stage-- > 0;) {
            if (owner[(size_t) stage] >= 0)
                continue;
            for (size_t g = 0; g < groups.size(); ++g) {
                if (next[g] != stage || std::find(owner.begin(), owner.end(), (int) g) != owner.end())
                    continue;
                if (!model.launch(stage, (int) g, err)) {
                    ok = false;
                    break;
                }
                owner[(size_t) stage] = (int) g;
                break;
            }
            if (!ok)
                break;
        }
        if (model.peak_in_flight)
            *model.peak_in_flight = std::max(*model.peak_in_flight,
                (int) std::count_if(owner.begin(), owner.end(), [](int g) { return g >= 0; }));
    }
    if (ok && model.finish)
        for (size_t group = 0; group < groups.size(); ++group)
            if (!groups[group].rows.empty() && !model.finish((int) group, groups[group], err))
                return false;
    return ok;
}

bool Step::run(const std::vector<Decoding*>& active, std::string& err) {
    const int count = model_.run_groups ? std::min(settings.groups, (int) active.size()) : 1;
    if (count <= 1) {
        Work work;
        const auto t0 = std::chrono::steady_clock::now();
        prepare(active, work);
        auto& g = work.group;
        if (g.rows.empty())
            return true;
        if (!model_.run(g.rows, g.picks, err))
            return false;
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        return complete(work, -1, ms, err);
    }
    std::vector<Work> work((size_t) count);
    group_turns_.resize((size_t) count);
    std::vector<StepGroup> groups;
    const int wave_conversations = (int) std::count_if(active.begin(), active.end(),
        [](const Decoding* d) { return !d->stop; });
    // Picture/suffix answers can differ across live group changes; see docs/DETAILS.md's validation limits.
    for (int i = 0; i < count; ++i) {
        const size_t first = active.size() * (size_t) i / count;
        const size_t end = active.size() * (size_t) (i + 1) / count;
        prepare({active.begin() + first, active.begin() + end}, work[(size_t) i], i, wave_conversations);
        groups.push_back(std::move(work[(size_t) i].group));
    }
    const auto t0 = std::chrono::steady_clock::now();
    if (!model_.run_groups(groups, err))
        return false;
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    for (int i = 0; i < count; ++i) {
        work[(size_t) i].group = std::move(groups[(size_t) i]);
        if (!work[(size_t) i].group.rows.empty() && !complete(work[(size_t) i], i, ms, err))
            return false;
    }
    return true;
}

void Step::prepare(const std::vector<Decoding*>& active, Work& work, int group, int wave_conversations) {
    std::vector<Decoding*> going;
    for (Decoding* d : active) {
        d->emitted.clear();
        d->fed.clear();
        d->finish = d->stop ? "cancel" : nullptr;
        if (!d->stop)
            going.push_back(d);
        else
            end_drafting(*d);
    }
    if (going.empty())
        return;
    if (model_.begin)
        model_.begin((int) going.size());
    const int ties = (int) (std::find_if(going.begin(), going.end(),
                                         [&](const Decoding* d) { return d->priority != going[0]->priority; }) -
                            going.begin());
    int& rotation = group < 0 ? turn_ : group_turns_[(size_t) group];
    const int turn = rotation % ties;
    rotation = turn + 1;
    const std::vector<int> share = share_guess_rows(settings, (int) going.size(), ties, turn);
    auto& rows = work.group.rows;
    auto& windows = work.windows;
    auto& stepping = work.stepping;
    auto& start = work.start;
    std::vector<int32_t> guesses((size_t) std::max(settings.rows, 1));
    const bool draft = settings.max_draft_conversations == 0 ||
                       std::max((int) going.size(), wave_conversations) <= settings.max_draft_conversations;
    const auto get_want = [&](size_t i) {
        const Decoding& d = *going[i];
        return !draft || !d.draft_enabled ? 0 :
            (int) std::max<int64_t>(std::min<int64_t>(share[i], d.max_new - d.produced - 1), 0);
    };
    std::vector<DraftRequest> proposals;
    if (model_.propose_batch) {
        proposals.resize(going.size());
        for (size_t i = 0; i < going.size(); ++i) {
            const Decoding& d = *going[i];
            auto& p = proposals[i];
            p.session = d.session;
            p.token = d.next;
            p.pos = d.pos;
            const int want = get_want(i);
            p.limit = want > 0 && d.draft_cooling ? 0 : settings.model_lookahead < 0 ? want :
                std::min(want, settings.model_lookahead);
            p.tokens.resize((size_t) std::max(settings.rows, 1));
            p.confidence.resize((size_t) DraftPolicy::kMaxT);
            p.context = {d.planner.get(), p.confidence.data(), d.draft_limits};
        }
        model_.propose_batch(proposals);
    }
    for (size_t i = 0; i < going.size(); ++i) {
        Decoding& d = *going[i];
        // no guess past what it may still emit, which also keeps its rows inside the context
        const int want = get_want(i);
        std::vector<float> probs((size_t) DraftPolicy::kMaxT);
        DraftContext context{d.planner.get(), probs.data(), d.draft_limits};
        const bool cooling = (model_.propose || model_.propose_batch) && want > 0 && d.draft_cooling;
        if (cooling)
            d.draft_cooling = false;
        const int model_want = cooling ? 0 : settings.model_lookahead < 0 ? want :
            std::min(want, settings.model_lookahead);
        int n = 0;
        if (model_.propose_batch) {
            auto& p = proposals[i];
            n = p.count;
            std::copy_n(p.tokens.begin(), n, guesses.begin());
            probs = p.confidence;
            context = p.context;
            context.probs = probs.data();
        } else if (model_.propose)
            n = model_.propose(d.session, d.next, d.pos, model_want, guesses.data(), &context);
        const int offered = n;
        n = count_confident_drafts(probs.data(), n, d.draft_limits.confidence, d.draft_limits.cumulative);
        if (offered > 0 && n == 0 && d.draft_limits.cool_after_rejection)
            d.draft_cooling = true;
        Window window{n + 1};
        if (d.planner) {
            auto& planner = *d.planner;
            if (planner.cost_aware && !context.cost_planned)
                for (int j = 0; j < n; ++j)
                    if (!context.keep_draft(j, probs[(size_t) j]))
                        break;
            window = planner.begin(d.pos, {.use_mtp = n > 0, .draft_on = n > 0,
                                           .first_window = d.produced == 0 || want == 0, .cap = n + 1,
                                           .start = n + 1, .dprob = probs.data()});
            window.T = std::min(window.T, n + 1);
            planner.pick_lookup(window, guesses.data(), false, want + 1);
            planner.pick_chain(window, guesses.data(), want + 1);
            if (window.lookup)
                std::copy_n(planner.lookup_drafts(), window.T - 1, guesses.data());
            if (window.chain > 0)
                std::copy_n(planner.chain_drafts(), window.chain, guesses.data() + window.T - 1);
            n = window.rows() - 1;
        }
        if (!model_.make_room(d.session, d.pos + 1 + n)) {
            // Optional guesses must not prevent the conversation's own next token from fitting.
            if (n == 0 || !model_.make_room(d.session, d.pos + 1)) {
                d.finish = "length";
                continue;
            }
            n = 0;
            window = Window{1};
        }
        windows.push_back(window);
        stepping.push_back(&d);
        start.push_back(rows.size());
        rows.push_back({d.session, d.next, d.pos, -1});
        for (int j = 0; j < n; ++j)
            rows.push_back({d.session, guesses[(size_t) j], d.pos + 1 + j, (int) rows.size() - 1});
    }
    work.group.picks.resize(rows.size());
}

bool Step::complete(Work& work, int group, double ms, std::string& err) {
    auto& rows = work.group.rows;
    auto& picks = work.group.picks;
    auto& stepping = work.stepping;
    auto& windows = work.windows;
    auto& start = work.start;
    const auto commit_start = std::chrono::steady_clock::now();
    std::vector<int> kept;
    for (size_t c = 0; c < stepping.size(); ++c) {
        const size_t end = c + 1 < stepping.size() ? start[c + 1] : rows.size();
        Decoding& d = *stepping[c];
        kept.push_back(accept(d, rows.data() + start[c], picks.data() + start[c], (int) (end - start[c])));
    }
    if (!(group < 0 ? model_.keep(rows, kept, err) : model_.keep_group(group, rows, kept, err)))
        return false;
    ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - commit_start).count();
    for (size_t c = 0; c < stepping.size(); ++c) {
        Decoding& d = *stepping[c];
        const size_t end = c + 1 < stepping.size() ? start[c + 1] : rows.size();
        if (!d.planner)
            continue;
        std::vector<int32_t> tokens;
        for (size_t r = start[c]; r < end; ++r)
            tokens.push_back(rows[r].token);
        const int accepted = d.planner->accept(windows[c], tokens.data(), picks.data() + start[c]);
        // This step's timing excludes later cache effects of speculative expert traffic.
        // Joint passes do not measure the cost of one conversation's window width.
        if (stepping.size() == 1 && group < 0)
            d.planner->observe(windows[c], accepted, ms, d.finish != nullptr);
        for (int32_t token : d.emitted)
            d.planner->append(token);
        d.suffix_offered = d.planner->lookup.drafts + d.planner->chained.drafts;
        d.suffix_kept = d.planner->lookup.ok + d.planner->chained.ok;
    }
    for (const Decoding* d : stepping)
        if (d->finish != nullptr)
            end_drafting(*d);
    return true;
}

int Step::accept(Decoding& d, const StepRow* rows, const int32_t* picks, int n) const {
    int k = 1;   // the rows whose tokens are the conversation's: its token, then the guesses the model picked too
    while (k < n && picks[k - 1] == rows[k].token)
        ++k;
    if (std::any_of(picks, picks + k, [](int32_t p) { return p < 0; })) {
        d.finish = "nonfinite";
        return 1;
    }
    int e = 0;
    while (e < k && d.finish == nullptr) {
        d.emitted.push_back(picks[e]);
        d.fed.push_back(rows[e].token);
        ++d.produced;
        if (is_end(picks[e]))
            d.finish = "stop";
        else if (d.produced >= d.max_new)
            d.finish = "length";
        ++e;
    }
    d.pos += e;
    d.next = picks[e - 1];
    d.guessed += n - 1;
    d.guessed_right += e - 1;
    return e;
}

void Step::end_drafting(const Decoding& d) const {
    if (model_.propose)
        (void) model_.propose(d.session, d.next, d.pos, 0, nullptr, nullptr);
}

bool Step::is_end(int32_t token) const { return std::find(end_.begin(), end_.end(), (int64_t) token) != end_.end(); }

}  // namespace strata::spec
