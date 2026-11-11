// src/models/deepseekv41/conversation_state.cpp - a DeepSeek conversation's host side and its bytes
// (conversation_state.hpp).
#include "strata/models/deepseekv41/conversation_state.hpp"
#include "strata/common/state/restore_points.hpp"
#include "strata/common/state/state_bytes.hpp"

#include <algorithm>
#include <cstring>
#include <utility>

namespace strata::ds4 {
namespace {

constexpr uint64_t kStateMagic = 0x3154415453345344ull;   // "DS4STAT1" (little endian)
/// The least bytes a checkpoint's layer takes: the counts of its ring and its partial group's two parts.
constexpr uint64_t kLayerBytesMin = 3 * sizeof(uint64_t);

using strata::core::append_state_bytes;
using strata::core::append_state_u64;
using strata::core::append_state_vector;
using strata::core::StateReader;

}  // namespace

std::vector<int64_t> get_check_positions(const ConversationState& s) {
    std::vector<int64_t> v;
    const auto add = [&](const Checkpoint& c) {
        if (c.pos > 0 && c.pos <= (int64_t) s.tokens.size() && !c.win.empty() &&
            std::find(v.begin(), v.end(), c.pos) == v.end())
            v.push_back(c.pos);
    };
    std::for_each(s.checks.begin(), s.checks.end(), add);
    std::for_each(s.marks.begin(), s.marks.end(), add);
    std::sort(v.rbegin(), v.rend());
    return v;
}

void encode_state(const ConversationState& s, std::vector<uint8_t>& out) {
    std::vector<std::pair<uint64_t, const Checkpoint*>> kept;   // (role, checkpoint)
    for (size_t role = 0; role < s.checks.size(); ++role)
        if (s.checks[role].pos >= 0)
            kept.push_back({role, &s.checks[role]});
    const size_t first_mark = kept.size();
    for (const Checkpoint& c : s.marks)
        kept.push_back({(uint64_t) kCheckpoints + c.shared, &c});
    std::stable_sort(kept.begin() + (std::ptrdiff_t) first_mark, kept.end(),
                     [](const auto& a, const auto& b) { return a.second->used < b.second->used; });
    out.clear();
    append_state_u64(out, kStateMagic);
    append_state_vector(out, s.tokens);
    append_state_u64(out, kept.size());
    for (const auto& [role, kept_check] : kept) {
        const Checkpoint& c = *kept_check;
        append_state_u64(out, role);
        append_state_bytes(out, &c.pos, sizeof c.pos);
        append_state_u64(out, c.win.size());
        for (size_t l = 0; l < c.win.size(); ++l) {
            append_state_vector(out, c.win[l]);
            for (size_t k = 2 * l; k < 2 * l + 2; ++k)
                append_state_vector(out, k < c.comp.size() ? c.comp[k] : std::vector<float>{});
        }
    }
    append_state_u64(out, kStateMagic);
}

bool decode_state(const std::vector<uint8_t>& in, ConversationState& s, std::string& err) {
    StateReader r{in};
    ConversationState got;
    bool ok = r.take_u64() == kStateMagic;
    r.take_vector(got.tokens);
    const uint64_t n = r.take_u64();
    ok = ok && r.ok && n <= (uint64_t) (kCheckpoints + strata::core::kMarksMax);
    for (uint64_t i = 0; ok && i < n; ++i) {
        const uint64_t role = r.take_u64();
        ok = r.ok && role <= (uint64_t) kCheckpoints + 1 &&
             (role >= (uint64_t) kCheckpoints || got.checks[(size_t) role].pos < 0);
        if (!ok)
            break;
        Checkpoint& c = role < (uint64_t) kCheckpoints ? got.checks[(size_t) role] : got.marks.emplace_back();
        c.shared = role == (uint64_t) kCheckpoints + 1;
        c.used = role < (uint64_t) kCheckpoints ? 0 : got.marks.size();
        const uint64_t layers = r.take(&c.pos, sizeof c.pos) ? r.take_u64() : 0;
        ok = r.ok && c.pos >= 0 && c.pos <= (int64_t) got.tokens.size() &&
             layers <= (in.size() - r.at) / kLayerBytesMin;
        c.win.resize(ok ? (size_t) layers : 0);
        c.comp.resize(c.win.size() * 2);
        for (size_t l = 0; ok && l < c.win.size(); ++l) {
            r.take_vector(c.win[l]);
            r.take_vector(c.comp[2 * l]);
            r.take_vector(c.comp[2 * l + 1]);
            ok = r.ok;
        }
    }
    if (!ok || r.take_u64() != kStateMagic || !r.ok || r.at != in.size()) {
        err = "the paused conversation's checkpoints are cut short or damaged";
        return false;
    }
    s = std::move(got);
    return true;
}

}  // namespace strata::ds4
