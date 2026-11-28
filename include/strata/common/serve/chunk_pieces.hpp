#pragma once
// strata/common/serve/chunk_pieces.hpp - the physical pieces a paged engine's GPU pool takes a chunk in, where the
// pieces' size decides how full the GPU's dedicated memory gets: an integrated GPU under Windows, with virtual memory
// (get_gpu_chunk_memory in src/common/serve/paged_start.cpp).

#include <cstdint>

namespace strata::serve {

/// Uses 16 MiB physical pieces for a paged GPU pool's chunk after a whole chunk spills into shared memory.
/// The Strix Halo comparison and its conditions are recorded in docs/ENGINE_NOTES.md.
constexpr uint64_t kPieceBytes = 16ull << 20;

/// Where a chunk's pieces landed: not taken (no memory), all in the GPU's dedicated memory, or some outside it.
enum class ChunkLanding { none, inside, outside };

/// Takes a chunk of `chunk_bytes` as one physical piece and, when that lands outside the dedicated memory, gives it
/// back and takes the chunk in pieces of kPieceBytes instead, so that the room the large piece did not fit in is used.
/// `take(piece_bytes)` takes the chunk in pieces of `piece_bytes` (the last one the rest of the chunk) and gets where
/// they landed (outside: any of them did); `give_back()` frees what `take` took.  Each chunk starts with one piece
/// again so room freed by other programs can be used as a whole piece.
template <class Take, class GiveBack>
ChunkLanding take_chunk_in_pieces(uint64_t chunk_bytes, Take take, GiveBack give_back) {
    const ChunkLanding whole = take(chunk_bytes);
    if (whole != ChunkLanding::outside || chunk_bytes <= kPieceBytes)
        return whole;
    give_back();
    return take(kPieceBytes);
}

}  // namespace strata::serve
