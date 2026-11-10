// src/models/qwen38/conversation.cpp - one sequence on a loaded Qwen3.8 model (conversation.hpp).
#include "strata/models/qwen38/conversation.hpp"

#include <algorithm>
#include <chrono>

namespace strata::qwen38 {

bool Conversation::init_prompt(Model& m, strata::core::SessionState& ss, void* borrow, uint64_t borrow_bytes,
                               std::string& err) {
    if (!m.multi_gpu && !m.o.no_pool) sp.set_cpu_pool(&*m.cpu_pool);
    return sp.init(m.wt, m.g, ss, m.srcp, m.o.expert_cache > 0 ? &m.xcache : nullptr,
                   m.host_res.empty() ? nullptr : m.host_res.data(), m.o.prefill_chunk, m.main_cs, err, borrow,
                   borrow_bytes);
}

bool Conversation::init_decode(Model& m, strata::core::SessionState& ss, int max_rows, std::string& err) {
    m_ = &m;
    if (!ver.init(m.wt, m.g, ss, m.get_verify_hits(), m.native_head.loaded() ? &m.native_head : nullptr, max_rows, err))
        return false;
    ver.set_split(m.o.spec_split);
    ver.set_pcie_mode(get_pcie_mode_code(m.o));
    m.drive.d.plan = ver.plan_sink();
    m.drive.d.pcie_num = get_pcie_num(m.o.pcie_frac);
    return true;
}

bool Conversation::run_window(int T, const int32_t* tokens, int64_t pos0, int32_t* out, std::string& err,
                              bool invariant) {
    return run_through_pool([&] { return ver.run(T, tokens, pos0, get_pool(invariant), get_pool_user(), out, err); },
                            err);
}

bool Conversation::run_rows(const int* sessions, int S, const int32_t* tokens, const int64_t* pos, int32_t* out,
                            std::string& err) {
    return run_through_pool(
        [&] { return ver.run_slot_rows(sessions, S, tokens, pos, get_pool(), get_pool_user(), out, err); }, err);
}

bool Conversation::run_through_pool(const std::function<bool()>& window, std::string& err) {
    strata::core::ExpertDispatch& d = m_->drive.d;
    d.layers = 0;
    d.experts = 0;
    d.failed = false;
    if (!window())
        return false;
    if (d.failed) {
        err = "the expert pool failed at layer " + std::to_string(d.fail_layer) + " expert " +
              std::to_string(d.fail_expert) + ": " + (d.fail ? d.fail : "(no message)");
        return false;
    }
    return true;
}

strata::kernels::SamplerParams get_sampler_params(const strata::serve::Sampling& s) {
    strata::kernels::SamplerParams sp;
    sp.greedy = s.temperature <= 0.0f;
    sp.temperature = s.temperature;
    sp.top_p = s.top_p;
    sp.top_k = s.top_k;
    sp.seed = s.seed ? s.seed : (unsigned long long) std::chrono::steady_clock::now().time_since_epoch().count();
    sp.min_p = std::clamp(s.min_p, 0.0f, 1.0f);
    sp.penalty_last_n = std::max(s.penalty_last_n, 0);
    sp.penalty_repeat = s.penalty_repeat;
    sp.penalty_freq = s.penalty_freq;
    sp.penalty_present = s.penalty_present;
    sp.counter = 0;
    return sp;
}

}  // namespace strata::qwen38
