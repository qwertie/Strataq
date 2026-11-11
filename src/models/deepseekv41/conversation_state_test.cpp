// A DeepSeek conversation's host side (conversation_state.hpp): its bytes' round trip and layout, every way damaged
// bytes are refused, and the positions it can go back to.  No GPU, no model files.
#include "strata/models/deepseekv41/conversation_state.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace strata::ds4;

namespace {

int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) {
        std::fprintf(stderr, "FAIL: %s\n", description);
        std::exit(1);
    }
}

/// Gets a checkpoint at `pos` of `layers` layers, every second one with a partial group, its bytes from `seed`.
Checkpoint make_check(int64_t pos, size_t layers, uint8_t seed) {
    Checkpoint c;
    c.pos = pos;
    for (size_t l = 0; l < layers; ++l) {
        c.win.emplace_back(64 + l, (uint8_t) (seed + l));
        c.comp.emplace_back(l % 2 ? 8 : 0, (float) seed + 0.5f);
        c.comp.emplace_back(l % 2 ? 8 : 0, -(float) seed);
    }
    return c;
}

bool is_same(const ConversationState& a, const ConversationState& b) {
    if (a.tokens != b.tokens)
        return false;
    for (int k = 0; k < kCheckpoints; ++k)
        if (a.checks[k].pos != b.checks[k].pos || a.checks[k].win != b.checks[k].win ||
            a.checks[k].comp != b.checks[k].comp)
            return false;
    if (a.marks.size() != b.marks.size())
        return false;
    for (size_t k = 0; k < a.marks.size(); ++k)
        if (a.marks[k].pos != b.marks[k].pos || a.marks[k].win != b.marks[k].win ||
            a.marks[k].comp != b.marks[k].comp || a.marks[k].shared != b.marks[k].shared ||
            a.marks[k].used != b.marks[k].used)
            return false;
    return true;
}

}  // namespace

int main() {
    ConversationState s;
    for (int32_t i = 0; i < 300; ++i)
        s.tokens.push_back(7000 + i);
    // a prompt's end at 300 and the position before it; the anchor, a copy of an earlier prompt's position before
    // its end; restore points at a shared start and at the last user message, least recently used first
    s.checks[0] = make_check(300, 5, 1);
    s.checks[1] = make_check(299, 5, 2);
    s.checks[2] = make_check(120, 5, 3);
    s.marks = {make_check(100, 5, 4), make_check(250, 5, 5)};
    s.marks[0].shared = true;
    s.marks[0].used = 1;
    s.marks[1].used = 2;
    std::vector<uint8_t> bytes;
    encode_state(s, bytes);
    ConversationState back;
    std::string err;
    check(decode_state(bytes, back, err) && is_same(s, back), "the round trip gives the same state");
    ConversationState swapped = s;   // (a mark clock elsewhere: the order is what comes back)
    std::swap(swapped.marks[0], swapped.marks[1]);
    swapped.marks[0].used = 70;
    swapped.marks[1].used = 40;
    encode_state(swapped, bytes);
    check(decode_state(bytes, back, err) && is_same(s, back), "restore points come back least recently used first");
    ConversationState no_anchor = s;
    no_anchor.checks[2] = Checkpoint{};
    encode_state(no_anchor, bytes);
    check(decode_state(bytes, back, err) && back.checks[2].pos == -1 && back.checks[2].win.empty(),
          "a checkpoint not kept stays none");
    encode_state(s, bytes);

    // the layout: magic, tokens, the checkpoint count, the first checkpoint's role, position and layer count
    uint64_t u = 0;
    std::memcpy(&u, bytes.data() + 8, 8);
    check(u == 300, "the token count follows the magic");
    int32_t t = 0;
    std::memcpy(&t, bytes.data() + 16 + 4 * 299, 4);
    check(t == 7299, "then the tokens");
    const size_t checks_at = 16 + 4 * 300;
    std::memcpy(&u, bytes.data() + checks_at, 8);
    check(u == 5, "then the count of kept checkpoints and restore points");
    int64_t pos = 0;
    std::memcpy(&u, bytes.data() + checks_at + 8, 8);
    std::memcpy(&pos, bytes.data() + checks_at + 16, 8);
    check(u == 0 && pos == 300, "each checkpoint starts with its role and position");
    std::memcpy(&u, bytes.data() + checks_at + 24, 8);
    check(u == 5, "and its layer count");
    std::memcpy(&u, bytes.data() + bytes.size() - 8, 8);
    uint64_t first = 0;
    std::memcpy(&first, bytes.data(), 8);
    check(u == first && std::memcmp(bytes.data(), "DS4STAT1", 8) == 0, "the magic at both ends");

    // refused: every cut length, a byte too many, a damaged magic at either end, a checkpoint past the tokens, a role
    // twice
    const ConversationState before = back;
    for (size_t keep = 0; keep < bytes.size(); ++keep) {
        const std::vector<uint8_t> cut(bytes.begin(), bytes.begin() + (std::ptrdiff_t) keep);
        if (decode_state(cut, back, err)) {
            std::fprintf(stderr, "FAIL: bytes cut to %zu of %zu were read\n", keep, bytes.size());
            return 1;
        }
    }
    check(is_same(back, before), "a refusal leaves the state as it was");
    std::vector<uint8_t> more = bytes;
    more.push_back(0);
    check(!decode_state(more, back, err), "a byte past the end is refused");
    std::vector<uint8_t> bad = bytes;
    bad[0] ^= 0x55;
    check(!decode_state(bad, back, err), "a damaged first magic is refused");
    bad = bytes;
    bad.back() ^= 0x55;
    check(!decode_state(bad, back, err), "a damaged last magic is refused");
    ConversationState past = s;
    past.checks[0].pos = 301;
    encode_state(past, bad);
    check(!decode_state(bad, back, err), "a checkpoint past the tokens is refused");
    bad = bytes;
    const uint64_t second_role = 1;
    std::memcpy(bad.data() + checks_at + 8, &second_role, 8);   // (the first checkpoint's role made the second's)
    check(!decode_state(bad, back, err), "a role kept twice is refused");
    check(!decode_state({}, back, err) && !err.empty(), "no bytes are refused, and say why");

    // the positions it can go back to: latest first, each once, none at 0 or past its tokens, none without rings
    check(get_check_positions(s) == std::vector<int64_t>({300, 299, 250, 120, 100}), "latest first");
    ConversationState d = s;
    d.checks[2] = d.checks[1];   // (Model::pin: the anchor at the position before the prompt's end)
    d.marks = {d.checks[1], make_check(0, 1, 6)};
    check(get_check_positions(d) == std::vector<int64_t>({300, 299}), "each once, none at 0");
    d.tokens.resize(299);
    d.checks[1].win.clear();
    d.checks[2].win.clear();
    d.marks.clear();
    check(get_check_positions(d).empty(), "none past the tokens or without rings");
    ConversationState empty;
    encode_state(empty, bytes);
    check(decode_state(bytes, back, err) && back.tokens.empty() && get_check_positions(back).empty(),
          "an empty conversation's round trip");
    std::printf("ds4_conversation_state_test: %d checks passed\n", checks);
    return 0;
}
