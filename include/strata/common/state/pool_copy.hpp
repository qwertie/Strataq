// include/strata/common/state/pool_copy.hpp - a conversation's pages copied between two page pools (a GPU's and the
// RAM's) and to and from a file on the SSD: how a conversation is paused and resumed (docs/MEMORY_MANAGER_PLAN.md,
// step M5, "Tiers and pausing").
#pragma once

#include "strata/common/state/page_pool.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace strata::core {

/// Selects bytes within each block of a region; bytes == 0 selects the rest of the block from `offset` on.
struct RegionRange {
    size_t region = 0;
    uint64_t offset = 0, bytes = 0;
};

/// The part of a pool's layout a copy takes: consecutive regions, after a share of a fixed region when `part` names
/// one.  strata-q38's RAM pool has the whole model's regions, of which one card of a layer split holds its share of
/// the GDN layers' state (a byte range of one fixed region) and its QSA layers' regions.
struct RegionSpan {
    size_t first = 0;
    size_t count = SIZE_MAX;   ///< SIZE_MAX: to the layout's end
    size_t part = SIZE_MAX;    ///< the fixed region whose share comes first; SIZE_MAX: none
    uint64_t offset = 0;       ///< where the share starts in the fixed region's block
    uint64_t bytes = 0;        ///< the share's size
    /// Additional pieces before the consecutive regions, for layouts with several packed fixed ranges.
    std::vector<RegionRange> ranges;
};

/// Copies conversation `from_id`'s blocks in pool `from` into conversation `to_id`'s in pool `to`: per piece of
/// `to_span` (the fixed share, explicit ranges, then consecutive regions), as many first blocks as `to` holds, from
/// the piece in the same place of `from_span`, which must hold them too. The spans have as many pieces, with matching
/// block sizes (a share's size), and the pools the same page size; either pool may be in a GPU's memory. The copies are done on
/// return.
bool copy_conversation(const PagePool& from, ConversationId from_id, const PagePool& to, ConversationId to_id,
                       std::string& err, RegionSpan from_span = {}, RegionSpan to_span = {});
/// Writes the blocks of conversation `id`'s pieces `span` in `pool` as one section at the current position of an open
/// binary stream (a pause file's, pause_file.hpp): per piece its block bytes and count, then the blocks.
bool write_conversation_section(std::FILE* file, const PagePool& pool, ConversationId id, std::string& err,
                                RegionSpan span = {});
/// Reads the section write_conversation_section wrote at the current position of an open binary stream into the blocks
/// of conversation `id`'s pieces `span` in `pool`: per piece, as many of the section's first blocks as the conversation
/// holds, which the section must hold too.  The stream is left after the section.
bool read_conversation_section(std::FILE* file, const PagePool& pool, ConversationId id, std::string& err,
                               RegionSpan span = {});
/// Gets the start of the names of this process's files of paused conversations in folder `dir`: "paused-", its
/// process id, the time now and "-", so that engines sharing the folder keep apart and find_stale_pause_files can
/// tell whose a file is.
std::string get_pause_file_prefix(const std::string& dir);
/// Finds regular pause files whose owning process no longer runs; a process that took the owner's id after the file's
/// prefix was made does not count (on Windows; elsewhere such files stay untouched).  `running`, when given,
/// receives those whose owner still runs.
std::vector<std::string> find_stale_pause_files(const std::string& dir, std::vector<std::string>* running = nullptr);

}  // namespace strata::core
