// Stores a paged conversation in one file, shared by automatic SSD pauses and named sessions.
//
// Format STRPAUS2 is little-endian: magic, header size, header bytes and their hash, then one page section per card
// (write_conversation_section), followed by a hash of all preceding bytes. The header holds the identity, tokens,
// rewind positions, model state and layer cut, then optional pictures and last-token Unix time.
// Hashes detect accidental damage; restore trusted files only.
#pragma once

#include "strata/common/state/conversation_file.hpp"
#include "strata/common/state/pool_copy.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {

/// What a paused conversation's file holds besides its pages.
struct PauseHeader {
    /// the pictures used to compute these pages: where each starts and a hash of its content; empty for text
    std::vector<ConversationImageKey> images;
    std::vector<int32_t> live;     ///< the tokens the conversation holds
    std::vector<int64_t> points;   ///< the positions short of its end it can go back to
    std::vector<uint8_t> state;    ///< the model's state beside the pages, in a form that does not depend on the cut
    /// the layer cut the sections were written at: the first layer of each card after the first (so one section more
    /// than it has numbers)
    std::vector<int64_t> cut;
    /// Unix seconds when tokens were last processed; zero means unknown.
    uint64_t last_tokens_at = 0;
};

/// One card's pages of a conversation, as a section of a pause file: where they are now.
struct PauseSection {
    const PagePool* pool = nullptr;
    ConversationId id = 0;
    RegionSpan span;
    int device = -1;  ///< GPU owning the section; negative for host memory.
};

/// An open file, closed when it goes out of scope.
class PauseStream : public std::unique_ptr<std::FILE, int (*)(std::FILE*)> {
public:
    PauseStream() : unique_ptr(nullptr, &std::fclose) {}
};

/// Writes the pause file `path`: `head`, bound to `identity`, then each of `sections`.  It is written beside `path`
/// under another name, flushed to the drive and renamed over it, so a failure preserves an existing destination.
/// Returns false with `err` on failure.
bool write_pause_file(const std::string& path, const SessionFileIdentity& identity, const PauseHeader& head,
                      const std::vector<PauseSection>& sections, std::string& err);
/// Copies a complete pause file through a temporary file, flushed and published as write_pause_file does.
bool copy_pause_file(const std::string& from, const std::string& to, std::string& err);
/// Opens the pause file `path` and reads its header into `head`, leaving `file` at the first section.  Gets none, or
/// the reason the file was refused: invalid for a file that is not a pause file, is damaged, was written for another
/// `identity`, or holds more than `max_tokens` tokens; io when it cannot be read.
SessionError open_pause_file(const std::string& path, const SessionFileIdentity& identity, uint64_t max_tokens,
                             PauseStream& file, PauseHeader& head, std::string& err);

}  // namespace strata::core
