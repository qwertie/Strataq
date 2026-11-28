// include/strata/base/platform/piece_reads.hpp - one batch of reads of scattered file ranges, merged
// into large requests.  Used by the expert stores of Qwen (FileExpertSource) and DeepSeek (ExpertFiles).
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::platform {

/// A range of a file to read and where its bytes go.
struct FilePiece {
    intptr_t file;    ///< opened with os.hpp's open_direct or open_par (Windows: overlapped)
    uint64_t off;     ///< first byte in the file (any alignment)
    uint64_t bytes;   ///< length (zero skips the piece)
    uint8_t* dst;     ///< receives the `bytes` bytes (any alignment)
};

/// Reads every piece (Windows: overlapped reads; Linux: the kernel's io_submit) as one
/// queue, each piece's range widened to whole sectors, and ranges of a file that are near each other merged into one
/// request.  Returns false (some dst may then hold partial bytes) when a read fails, a range runs past the end of
/// its file, or a range is too large for the OS, so the caller can read the pieces another way. The handle's open
/// mode decides whether reads bypass the file cache.
/// Safe from several threads: each keeps its own staging buffer until the thread exits.
bool read_pieces(const FilePiece* pieces, size_t n);

}  // namespace strata::platform
