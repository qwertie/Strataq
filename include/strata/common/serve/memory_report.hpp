// strata/common/serve/memory_report.hpp - --serve's GET_MEMORY_USE: what the engine holds in each memory (each GPU,
// the RAM, the SSD), for the server's GET /v1/memory and the web page's Memory panel (docs/MEMORY_MANAGER_PLAN.md,
// step M0); and CHANGE_SETTINGS, which changes the settings it reports while the engine runs (step M6).
//
//   GET_MEMORY_USE
//     MEMORY <id> [total=<B>] [free=<B>] [unified=1] [<use>=<B> ...] [paused_count=<n>] [paused_budget=<B>]
//            [pool_pages=<n> allocated_pages=<n> fragmented_pages=<n>] [setting=<B> smallest=<B> largest=<B>]
//            [mapped=<B>] [name=<text>]                                                   one line per memory, then
//     [MEMORY_BLOCK <memory-id> <bytes> <category> <pool> <label> [<identity-group> <ids>]] labelled allocations
//     [PAUSE_DIRECTORY <path>]                                                  folder for automatic SSD pauses
//     [CONVERSATION <id> <place> <state> <tokens> <capacity> <bytes> <used>]     with per_conversation: one per
//                                                                                conversation (LIST_CONVERSATIONS')
//     SETTINGS [context=<tokens>] [concurrent=<n>] [rows=<n>] [lookahead=<n>] [tokens_min=<tokens>]
//     [SHORTFALL <text>]                                                         CHANGE_SETTINGS' answer only
//     END_MEMORY
//
// MEMORY_BLOCK fields are percent-encoded UTF-8; ~ denotes an empty label. Its bytes are disjoint: a pool's
// backing allocation is replaced by its labelled blocks and unused pages. A memory with MEMORY_BLOCK lines
// derives its category totals from those lines; it does not also send <use>= values on its MEMORY line.
//
// id: `gpu<N>` (the runtime's device number), `ram` or `ssd`.  total, free: the memory's size and what is free in it
// now (the GPUs'; the server adds the machine's RAM).  unified=1: a GPU whose memory is the system RAM (an integrated
// GPU such as Strix Halo's).  <use>: the bytes the engine holds there for one use - `weights` (the dense weights),
// `experts` (trunk), `drafter_experts` (draft model), `conversations` (the conversations it holds: sessions
// allocated for their whole context, or pool pages),
// `paused` (parked conversations), `drafter`, `scratch`, `engram` (DeepSeek's n-gram table).  paused_count: the
// parked conversations there; paused_budget: the most they may take.  pool_pages: the pages of a page pool there
// (strata-q38's), of which allocated_pages hold experts or conversations and fragmented_pages are free in runs
// shorter than the smallest expert, which no expert can use.  setting: the bytes the engine's pool there takes now
// (its `memory_mib.<id>` setting), and smallest, largest: the range CHANGE_SETTINGS accepts for it now.  mapped: the
// bytes of it that are page-locked for the GPUs to read (strata-q38's RAM pool); Windows counts them in a GPU's shared
// use too, so a reader takes them off that.  name: the device's name, to the end of the line.  settings: the context
// in tokens, the conversations decoded at once, the rows one step computes, the most guesses one conversation adds in
// a step, the conversation tokens kept free of experts, and what else the engine can change.  A value the engine does
// not know is left out, not guessed.  The command changes nothing.
//
//   CHANGE_SETTINGS <name>=<value> [...]      -> the GET_MEMORY_USE answer once changed, or ERR <why>, changing
//                                                nothing of the line
//
// A change the engine accepted can still get less than asked (a memory had less to give than it said); the answer
// then has a SHORTFALL line saying what, and its MEMORY lines show the sizes reached.
//
// The names are those of SETTINGS, and `memory_mib.<id>` for a memory's `setting` in MiB; what each takes is the
// engine's (strata-q38's: paged_engine.hpp).
#pragma once

#include "strata/common/state/token_listing.hpp"
#include "strata/base/platform/memory_manager.hpp"

#include <cstdint>
#include <deque>
#include <string>
#include <utility>
#include <vector>

namespace strata::serve {

/// One memory the engine uses: a GPU, the RAM or the SSD (a `MEMORY` line).
struct Memory {
    std::string id;                                       ///< gpu<N>, ram or ssd
    std::string name;                                     ///< the device's name; empty: not said
    uint64_t total = 0;                                   ///< its size in bytes; 0: unknown
    uint64_t free = 0;                                    ///< what is free in it now (said only with total)
    bool unified = false;                                 ///< a GPU whose memory is the system RAM
    std::vector<std::pair<std::string, uint64_t>> uses;   ///< bytes held per use, in the order first added
    std::vector<core::MemoryBlock> blocks;   ///< disjoint allocation totals with pool and purpose labels
    uint64_t paused_budget = 0;                           ///< the most parked conversations may take; 0: no budget
    int paused_count = 0;                                 ///< parked conversations held here
    /// The pages of the page pool here (0: none; the old engines'), how many are allocated, and how many are free in
    /// runs too short for any expert (PagePoolReport::allocated, fragmented).
    uint64_t pool_pages = 0, allocated_pages = 0, fragmented_pages = 0;
    /// The bytes the engine's pool here takes (0: no setting to change), and the least and most it may be set to now.
    uint64_t setting = 0, smallest = 0, largest = 0;
    uint64_t pool_offset = 0;   ///< bytes drawn before the adjustable pool, independent of category names
    uint64_t mapped = 0;   ///< the bytes of it the GPUs map (page-locked); 0: none or not said

    /// Adds `bytes` to use `use`; a zero adds nothing.
    void add(const std::string& use, uint64_t bytes) {
        if (bytes == 0) return;
        for (auto& [u, b] : uses)
            if (u == use) {
                b += bytes;
                return;
            }
        uses.emplace_back(use, bytes);
    }
    /// Includes labelled blocks and derives the category totals from them.
    void add_blocks(const std::vector<core::MemoryBlock>& entries) {
        for (const core::MemoryBlock& b : entries) {
            add(b.name.category, b.bytes);
            core::add_memory_block(blocks, b.name, b.bytes, b.identity_group, b.ids);
        }
    }
};

/// What an engine says about its memories and settings (EngineHooks::describe_memory).
struct MemoryReport {
    std::deque<Memory> memories;   ///< (a deque: at() keeps the references it gave out valid)
    std::vector<std::pair<std::string, int64_t>> settings;   ///< SETTINGS' keys, in order
    /// Each conversation the engine holds gets a CONVERSATION line (what it takes, conversation by conversation).
    bool per_conversation = false;
    /// CHANGE_SETTINGS' answer: what the change got less of than asked, in words (a SHORTFALL line); empty: nothing.
    std::string shortfall;
    /// Folder used by this engine for automatic SSD pause files.
    std::string pause_directory;
    /// Supplies UI labels and discovery information without putting strings in numeric SETTINGS.
    std::vector<std::pair<std::string, std::string>> details;

    /// Gets the memory named `id`, adding an empty one at the end when there is none.
    Memory& at(const std::string& id) {
        for (Memory& m : memories)
            if (m.id == id) return m;
        Memory& m = memories.emplace_back();
        m.id = id;
        return m;
    }
    /// Gets GPU `device`'s memory (at("gpu<device>")).
    Memory& gpu(int device) { return at("gpu" + std::to_string(device)); }
};
/// Encodes a UTF-8 protocol field without limiting the category or label vocabulary.
inline std::string encode_memory_label(const std::string& text) {
    const char* hex = "0123456789ABCDEF";
    std::string out;
    for (const unsigned char c : text) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-')
            out += (char) c;
        else {
            out += '%';
            out += hex[c >> 4];
            out += hex[c & 15];
        }
    }
    return out.empty() ? "~" : out;
}

/// Writes the answer to GET_MEMORY_USE: `report` plus the parked conversations of `held` (LIST_CONVERSATIONS' entries;
/// a parked one in RAM counts on `ram`, one on disk on `ssd`).
inline std::string format_memory_report(MemoryReport report, const std::vector<core::HeldConversation>& held) {
    for (const core::HeldConversation& c : held) {
        if (std::string(c.state) != "parked") continue;
        Memory& m = report.at(std::string(c.place) == "disk" ? "ssd" : "ram");
        if (!m.pool_pages || m.blocks.empty()) {
            if (m.blocks.empty())
                m.add("paused", c.bytes);
            else
                m.add_blocks({{{"paused", "Paused conversations", c.id}, c.bytes}});
        }
        ++m.paused_count;
    }
    std::string s;
    const auto put = [&s](const std::string& key, uint64_t v) { s += " " + key + "=" + std::to_string(v); };
    for (const Memory& m : report.memories) {
        s += "MEMORY " + m.id;
        if (m.total > 0) {
            put("total", m.total);
            put("free", m.free);
        }
        if (m.unified) put("unified", 1);
        if (m.blocks.empty())
            for (const auto& [use, bytes] : m.uses)
                put(use, bytes);
        if (m.paused_count > 0 || m.paused_budget > 0) put("paused_count", (uint64_t) m.paused_count);
        if (m.paused_budget > 0) put("paused_budget", m.paused_budget);
        if (m.pool_pages > 0) {
            put("pool_pages", m.pool_pages);
            put("allocated_pages", m.allocated_pages);
            put("fragmented_pages", m.fragmented_pages);
        }
        if (m.setting > 0) {
            put("setting", m.setting);
            if (!m.blocks.empty())
                put("pool_offset", m.pool_offset);
            put("smallest", m.smallest);
            put("largest", m.largest);
        }
        if (m.mapped > 0) put("mapped", m.mapped);
        if (!m.name.empty()) s += " name=" + m.name;
        s += '\n';
        for (const core::MemoryBlock& b : m.blocks) {
            s += "MEMORY_BLOCK " + m.id + " " + std::to_string(b.bytes) + " " +
                 encode_memory_label(b.name.category) + " " + encode_memory_label(b.name.pool) + " " +
                 encode_memory_label(b.name.label);
            if (!b.identity_group.empty() && !b.ids.empty()) {
                s += " " + encode_memory_label(b.identity_group) + " ";
                for (size_t i = 0; i < b.ids.size(); ++i) {
                    if (i)
                        s += ',';
                    s += std::to_string(b.ids[i]);
                }
            }
            s += '\n';
        }
    }
    if (report.per_conversation)
        for (const core::HeldConversation& c : held)
            s += core::format_conversation_line(c) + '\n';
    if (!report.pause_directory.empty())
        s += "PAUSE_DIRECTORY " + report.pause_directory + "\n";
    for (const auto& [key, value] : report.details)
        s += "MEMORY_DETAIL " + key + " " + encode_memory_label(value) + "\n";
    s += "SETTINGS";
    for (const auto& [key, v] : report.settings)
        s += " " + key + "=" + std::to_string(v);
    s += '\n';
    if (!report.shortfall.empty())
        s += "SHORTFALL " + report.shortfall + '\n';
    s += "END_MEMORY\n";
    return s;
}

/// Reads a `CHANGE_SETTINGS <name>=<value> ...` line into `values`, in order; false (and `err`) when a field has no
/// `=` or there is none.
inline bool parse_settings_line(const std::string& line, std::vector<std::pair<std::string, std::string>>& values,
                                std::string& err) {
    values.clear();
    for (size_t at = line.find(' '); at != std::string::npos;) {
        const size_t from = at + 1;
        at = line.find(' ', from);
        const std::string field = line.substr(from, at == std::string::npos ? std::string::npos : at - from);
        if (field.empty())
            continue;
        const size_t eq = field.find('=');
        if (eq == std::string::npos || eq == 0) {
            err = "CHANGE_SETTINGS takes <name>=<value> fields, not \"" + field + "\"";
            return false;
        }
        values.emplace_back(field.substr(0, eq), field.substr(eq + 1));
    }
    if (values.empty())
        err = "CHANGE_SETTINGS names no setting";
    return !values.empty();
}

}  // namespace strata::serve
