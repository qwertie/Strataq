// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// ds4_checkpoints.cu - the conversation's state at a position: the checkpoints, the restore points a request asks for,
// the way back to one of them (restore) and another conversation's state taking this one's place
// (swap_conversation).
#include "ds4_model_impl.hpp"

namespace strata::ds4 {

void Model::Impl::reset() {
    pos = 0;
    toks.clear();
    ctoks.clear();
    for (Check& c0 : ck) c0.pos = -1;
    mk.clear();
    for (Layer& l0 : L)
        for (int m = 0; m < l0.nm; ++m) {
            Layer& l = member(l0, m);
            Dev& d = devs[(size_t) l.dev];
            cudaSetDevice(d.id);
            if (l.win) cudaMemsetAsync(l.win, 0, (size_t) g.win * win_rb(), d.st);
            if (l.c_kv) cudaMemsetAsync(l.c_kv, 0, (size_t) l.ratio * g.hd * 4, d.st);
            if (l.c_score) cudaMemsetAsync(l.c_score, 0, (size_t) l.ratio * g.hd * 4, d.st);
        }
    for (Dev& d : devs) {
        cudaSetDevice(d.id);
        cudaStreamSynchronize(d.st);
    }
}

void Model::Impl::snap_into(Check& c) {
    c.pos = pos;
    c.win.resize(L.size());
    c.comp.resize(L.size() * 2);
    for (size_t i = 0; i < L.size(); ++i) {
        Layer& l = L[i];
        Dev& d = devs[(size_t) l.dev];
        cudaSetDevice(d.id);
        c.win[i].resize(get_checkpoint_ring_bytes(i));
        cudaMemcpyAsync(c.win[i].data(), l.win, c.win[i].size(), cudaMemcpyDeviceToHost, d.st);
        if (l.c_kv) {
            c.comp[2 * i].resize((size_t) l.ratio * g.hd);
            c.comp[2 * i + 1].resize((size_t) l.ratio * g.hd);
            cudaMemcpyAsync(c.comp[2 * i].data(), l.c_kv, c.comp[2 * i].size() * 4, cudaMemcpyDeviceToHost, d.st);
            cudaMemcpyAsync(c.comp[2 * i + 1].data(), l.c_score, c.comp[2 * i + 1].size() * 4,
                            cudaMemcpyDeviceToHost, d.st);
        }
    }
    for (Dev& d : devs) {
        cudaSetDevice(d.id);
        cudaStreamSynchronize(d.st);
    }
}

int Model::Impl::marks_cap() {
    static const int n = [] {
        const char* v = std::getenv("STRATA_DS4_RESTORE_POINTS");
        if (v == nullptr) v = std::getenv("DS4_RESTORE_POINTS");
        return v != nullptr ? std::clamp(std::atoi(v), 0, kMarksMax) : 4;
    }();
    return n;
}

std::vector<int64_t> Model::Impl::check_positions() const {
    std::vector<int64_t> v;
    for (int k : slot_checks())
        if (std::find(v.begin(), v.end(), check_at(k).pos) == v.end()) v.push_back(check_at(k).pos);
    std::sort(v.rbegin(), v.rend());
    return v;
}

std::vector<int> Model::Impl::slot_checks(int64_t upto) const {
    if (upto < 0) upto = pos;
    std::vector<int> v, m;
    for (int i = 0; i < kChecks + (int) mk.size(); ++i) {
        const Check& c = check_at(i);
        if (c.pos > 0 && c.pos <= upto && c.win.size() == L.size()) (i < kChecks ? v : m).push_back(i);
    }
    std::stable_sort(m.begin(), m.end(), [this](int a, int b) { return check_at(a).used < check_at(b).used; });
    v.insert(v.end(), m.begin(), m.end());
    return v;
}

bool Model::Impl::restore(int64_t limit) {
    const int b = best_check(limit);
    if (b < 0 || !get_check_problem(check_at(b)).empty()) return false;
    const Check& c = check_at(b);
    for (size_t i = 0; i < L.size(); ++i)
        for (int m = 0; m < L[i].nm; ++m) {   // (a group's members hold the same state: the lead's copy)
            Layer& l = member(L[i], m);
            Dev& d = devs[(size_t) l.dev];
            cudaSetDevice(d.id);
            cudaMemcpyAsync(l.win, c.win[i].data(), c.win[i].size(), cudaMemcpyHostToDevice, d.st);
            if (l.c_kv) {
                cudaMemcpyAsync(l.c_kv, c.comp[2 * i].data(), c.comp[2 * i].size() * 4, cudaMemcpyHostToDevice, d.st);
                cudaMemcpyAsync(l.c_score, c.comp[2 * i + 1].data(), c.comp[2 * i + 1].size() * 4,
                                cudaMemcpyHostToDevice, d.st);
            }
        }
    for (Dev& d : devs) {
        cudaSetDevice(d.id);
        cudaStreamSynchronize(d.st);
    }
    if (draft_taps != nullptr && drafter_loaded) {
        std::string err;
        if (!reseed_drafter_from_taps(c.pos, err))
            return false;
    }
    pos = c.pos;
    toks.resize((size_t) pos);
    ctoks.resize((size_t) pos);
    for (Check& o2 : ck)   // a later checkpoint saw tokens the state no longer has
        if (o2.pos > pos) o2.pos = -1;
    mk.erase(std::remove_if(mk.begin(), mk.end(), [this](const Check& x) { return x.pos > pos; }), mk.end());
    strata::core::touch_restore_point(mk, mark_clock, pos);
    return true;
}

void Model::Impl::set_position() {
    pos = (int64_t) toks.size();
    ctoks.resize((size_t) pos);
    for (size_t i = 0; i < (size_t) pos; ++i)
        ctoks[i] = toks[i] < pk.eg_vocab ? pk.eg_map[(size_t) toks[i]] : 0;
}

std::string Model::Impl::get_state_problem(const ConversationState& s) const {
    for (const Check& c : s.checks)
        if (c.pos >= 0)
            if (std::string why = get_check_problem(c); !why.empty()) return why;
    for (const Check& c : s.marks)
        if (std::string why = get_check_problem(c); !why.empty()) return why;
    return {};
}

std::string Model::Impl::get_check_problem(const Check& c) const {
    if (c.win.size() != L.size() || c.comp.size() != 2 * L.size())
        return "a checkpoint of " + std::to_string(c.win.size()) + " layers, not " + std::to_string(L.size());
    for (size_t i = 0; i < L.size(); ++i) {
        const size_t group = L[i].kv_src && L[i].ratio > 1 ? (size_t) L[i].ratio * g.hd : 0;
        if (c.win[i].size() != get_checkpoint_ring_bytes(i) || c.comp[2 * i].size() != group ||
            c.comp[2 * i + 1].size() != group)
            return "a checkpoint whose layer " + std::to_string(i) + " is not this model's";
    }
    return {};
}

void Model::Impl::swap_conversation(ConversationState& s) {
    for (Dev& d : devs) {
        cudaSetDevice(d.id);
        cudaStreamSynchronize(d.st);
    }
    toks.swap(s.tokens);
    for (int k = 0; k < kChecks; ++k)
        std::swap(ck[k], s.checks[(size_t) k]);
    mk.swap(s.marks);
    set_position();
}

std::string Model::Impl::slot_identity() const {
    const auto size_of = [](const std::string& p) {
        std::error_code ec;
        const auto s = std::filesystem::file_size(p, ec);
        return ec ? 0ull : (unsigned long long) s;
    };
    std::ostringstream id;
    id << "strata-ds4-session-4 engine=" << STRATA_VERSION << " mtp=" << g.n_mtp << " dim=" << g.dim << " hd=" << g.hd
       << " win=" << g.win << " ratios=";
    for (int r : g.ratios) id << r << ",";
    id << " kv=";
    for (int v : g.kv_src) id << v << ",";
    id << " ix=";
    for (int v : g.ix_src) id << v << ",";
    id << " ds4.txt:" << size_of(pk.dir + "/ds4.txt") << " engram.bin:" << size_of(pk.dir + "/engram.bin")
       << " tensors.txt:" << size_of(pk.dir + "/tensors.txt");
    for (size_t i = 0; i < pk.files.size(); ++i)
        id << " " << pk.files[i] << ":" << platform::file_size(pk.fds[i]);
    return id.str();
}

}  // namespace strata::ds4
