// --serve's LIST_CONVERSATIONS and GET_CONTEXT_WINDOWS (strata_server/server.py's API monitor shows them): the conversations
// the engine holds - in a session in VRAM, or parked in RAM or on disk - and the tokens of the ones asked for.  Each
// engine path (generate.cpp, q36_engine.cu, ds4_main.cpp) describes what it holds as HeldConversation entries; the
// commands are parsed and answered here, so the lines are the same on every path.
//
//   LIST_CONVERSATIONS [<marker> <n>]
//     CONVERSATION <id> <place> <state> <tokens> <capacity> <bytes> <used> [last_tokens_at=<seconds>] [<preview>]
//     END_CONVERSATIONS
//   GET_CONTEXT_WINDOWS <id> [<id> ...]
//     CONTEXT_WINDOW <id> <n> [<ids>]   or   CONTEXT_WINDOW <id> gone   one line each, then END_CONTEXT_WINDOWS
//
// id: `main` (the engine's own session), `slot<b>` (a --batch slot's session) or `parked<serial>` (a parked
// conversation; the serial is never reused).  place: vram, ram or disk.  state: holding, decoding, reading (a prompt
// read that gave way and goes on later), empty or parked.  tokens: what it holds; capacity: the most it can hold
// (a session's --max-context; a parked conversation's own tokens).  bytes: its physical size (a session's allocation,
// a parked conversation's snapshot); used: what its tokens take of that (for a session the size a session sized for
// just them would have: its state and buffers and their K/V; 0 when empty).  Both bytes are 0 where the engine path
// does not know them (DeepSeek V4.1's session).  Token ids are comma-separated, as GEN takes them, and left out when
// there are none.  <marker> (comma-separated ids) and <n> ask for a preview: the first n tokens after the first place
// the conversation holds the marker - the server passes the chat template's user-turn header, so the preview is the
// start of the first user message.  Neither command changes anything.
#pragma once

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

namespace strata::core {

struct HeldConversation {
    std::string id;
    const char* place = "vram";
    const char* state = "holding";
    const int32_t* ids = nullptr;
    size_t n = 0;              // the tokens it holds
    uint64_t capacity = 0;     // the most it can hold, in tokens
    uint64_t bytes = 0;        // its physical size
    uint64_t used = 0;         // what its tokens take of that
    /// Unix seconds when tokens were last processed; zero means unknown.
    uint64_t last_tokens_at = 0;
};

inline bool is_conversation_command(const std::string& line) {
    return line.rfind("LIST_CONVERSATIONS", 0) == 0 || line.rfind("GET_CONTEXT_WINDOWS", 0) == 0;
}

namespace token_listing_detail {
inline void append_ids(std::string& s, const int32_t* ids, size_t n) {
    char num[16];
    for (size_t i = 0; i < n; ++i) {
        s += i == 0 ? ' ' : ',';
        const auto r = std::to_chars(num, num + sizeof num, ids[i]);
        s.append(num, r.ptr);
    }
}
inline bool parse_ids(const std::string& field, std::vector<int32_t>& out) {
    const char* p = field.data();
    const char* end = p + field.size();
    while (p < end) {
        int32_t v = 0;
        const auto r = std::from_chars(p, end, v);
        if (r.ec != std::errc() || (r.ptr != end && *r.ptr != ',')) return false;
        out.push_back(v);
        p = r.ptr == end ? end : r.ptr + 1;
    }
    return !out.empty();
}
inline void put(const std::string& s) { std::fwrite(s.data(), 1, s.size(), stdout); }
}  // namespace token_listing_detail

/// Gets LIST_CONVERSATIONS' line for `c`, without its preview and its end of line:
/// `CONVERSATION <id> <place> <state> <tokens> <capacity> <bytes> <used> [last_tokens_at=<seconds>]`.
inline std::string format_conversation_line(const HeldConversation& c) {
    return "CONVERSATION " + c.id + " " + c.place + " " + c.state + " " + std::to_string(c.n) + " " +
           std::to_string(c.capacity) + " " + std::to_string(c.bytes) + " " + std::to_string(c.used) +
           (c.last_tokens_at ? " last_tokens_at=" + std::to_string(c.last_tokens_at) : "");
}

/// Answers `line` (is_conversation_command) from `held`. A malformed command answers `ERR <why>`.
/// The optional sink lets a transport frame each complete reply line.
inline void answer_conversation_command(const std::string& line, const std::vector<HeldConversation>& held,
                                        void (*put)(const std::string&) = token_listing_detail::put) {
    namespace d = token_listing_detail;
    std::istringstream in(line);
    std::string command, field;
    in >> command;
    if (command == "LIST_CONVERSATIONS") {
        std::vector<int32_t> marker;
        size_t want = 0;
        if (in >> field) {
            std::string count;
            if (!d::parse_ids(field, marker) || !(in >> count) ||
                std::from_chars(count.data(), count.data() + count.size(), want).ec != std::errc()) {
                put("ERR LIST_CONVERSATIONS [<marker ids> <preview tokens>]\n");
                std::fflush(stdout);
                return;
            }
        }
        for (const HeldConversation& c : held) {
            std::string s = format_conversation_line(c);
            if (!marker.empty() && want > 0) {
                const int32_t* at = std::search(c.ids, c.ids + c.n, marker.begin(), marker.end());
                if (at != c.ids + c.n) {
                    at += marker.size();
                    d::append_ids(s, at, std::min(want, (size_t) (c.ids + c.n - at)));
                }
            }
            s += '\n';
            put(s);
        }
        put("END_CONVERSATIONS\n");
    } else if (command == "GET_CONTEXT_WINDOWS") {
        while (in >> field) {
            const auto c = std::find_if(held.begin(), held.end(), [&](const HeldConversation& h) { return h.id == field; });
            std::string s = "CONTEXT_WINDOW " + field;
            if (c == held.end()) {
                s += " gone";
            } else {
                s += " " + std::to_string(c->n);
                d::append_ids(s, c->ids, c->n);
            }
            s += '\n';
            put(s);
        }
        put("END_CONTEXT_WINDOWS\n");
    } else {
        put("ERR unknown command\n");
    }
    std::fflush(stdout);
}

}  // namespace strata::core
