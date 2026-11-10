// include/strata/models/deepseekv41/conversation_state.hpp - the part of a DeepSeek V4.1 conversation's state that
// the model keeps on the host, beside its window rings, partial groups and compressed rows: its tokens and its
// checkpoints (host copies of the rings and partial groups at a few positions, which it can go back to), and the
// bytes strata-ds41 keeps them in while the conversation is paused (docs/MEMORY_MANAGER_PLAN.md, step D6).  No GPU.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::ds4 {

/// The checkpoints a conversation keeps, by role (Model::checkpoint's `slot`): 0 its last prompt's end, 1 the position
/// before it, 2 the anchor (Model::pin), 3 the shared start of a prompt (strata-ds41's pin=N; the old engine keeps a
/// shared start as a restore point instead, Model::mark).
constexpr int kCheckpoints = 4;

/// The state at one position: what the rest of the state (compressed rows, by position) does not give back.
struct Checkpoint {
    int64_t pos = -1;                          ///< -1: none kept
    std::vector<std::vector<uint8_t>> win;     ///< Per layer: its ring; paged DSpark appends taps to the last ring.
    /// per layer, its compressor's partial group: its KV at 2 * layer, its scores at 2 * layer + 1 (empty for a layer
    /// without a compressor)
    std::vector<std::vector<float>> comp;
    bool shared = false;   ///< (a restore point Model::mark keeps) the end of a shared start, given up last
    uint64_t used = 0;     ///< (a restore point) its last use, by the model's mark clock
};

/// A conversation's host side: what the model holds of the conversation it computes, and what each other
/// conversation strata-ds41 holds keeps while the model computes another one (Model::swap_conversation).
struct ConversationState {
    std::vector<int32_t> tokens;                      ///< the tokens its state holds, in order
    std::array<Checkpoint, kCheckpoints> checks;      ///< by role
};

/// Gets the positions `s` can go back to (its checkpoints between its first and last token), latest first, each once:
/// what Model::check_positions gets for the conversation in the model.
std::vector<int64_t> get_check_positions(const ConversationState& s);
/// Writes `s` to `out` (replacing what it held).  The layout, little endian: u64 magic, u64 token count, i32 tokens,
/// u64 checkpoint count, then per kept checkpoint u64 role, i64 position, u64 layer count and per layer its ring
/// (u64 bytes, bytes), its partial group's KV and scores (u64 floats, floats each); u64 magic.
void encode_state(const ConversationState& s, std::vector<uint8_t>& out);
/// Reads encode_state's bytes `in` into `s`; false (and `err`, `s` unchanged) when they are cut short, damaged or not
/// such bytes.
bool decode_state(const std::vector<uint8_t>& in, ConversationState& s, std::string& err);

}  // namespace strata::ds4
