// strata_ds4_launch(kernel, grid, block, shared, stream, args...): the engine's one spelling of a kernel launch,
// kernel<<<grid, block, shared, stream>>>(args...) with an optional check (DS4_LAUNCH_CHECK).  (Every file that launches
// kernels includes this header.)
#pragma once
#if (defined(__CUDACC__) || defined(__HIPCC__) || defined(__HIP__))
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
template <class... KA, class... CA>
inline void strata_ds4_launch_at(const char* file, int line, void (*kernel)(KA...), dim3 grid, dim3 block, size_t shared,
                                 cudaStream_t stream, CA&&... args) {
    kernel<<<grid, block, shared, stream>>>(static_cast<KA>(args)...);
    static const bool check = std::getenv("DS4_LAUNCH_CHECK") != nullptr;   // report the launch that failed (not a later call)
    if (check) {
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess)
            std::fprintf(stderr, "ds4 launch failed at %s:%d: %s (grid %u,%u,%u block %u,%u,%u shared %zu)\n", file, line,
                         cudaGetErrorString(e), grid.x, grid.y, grid.z, block.x, block.y, block.z, shared);
    }
}
#define strata_ds4_launch(...) strata_ds4_launch_at(__FILE__, __LINE__, __VA_ARGS__)

// Gets the host address of device memory: `p` itself, so only the ds4_dbg dumps in ds4_model.cu (compiled out) call it.
namespace strata::gpu {
inline void* host_view(const void* p) { return const_cast<void*>(p); }
}
#endif

// ds4_dbg("NAME"): diagnostics written for a unified-memory GPU (DS4_DBG_*, DS4_CPU_LOG, DS4_TLOG, ...): the constant false, so
// the CUDA and HIP engines compile them out.  Several read device memory from the host (strata::gpu::host_view), which a
// discrete card does not allow, so they must not be switched on as they are.  TODO: delete them, or port them to copies.
constexpr bool ds4_dbg(const char*) { return false; }
