#include "strata/common/kernels/window_ops.hpp"
#include "strata/common/kernels/pdl.hpp"
#include "strata/base/gpu/emulate.hpp"
#include "strata/base/gpu/sm120_series.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <utility>
#include <vector>

namespace strata::kernels {
namespace {
void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
__global__ void copy_or_zero_kernel(float4* __restrict__ dst, const volatile float4* src, long long n4,
                                    const uint32_t* skip, uint32_t value) {
    const bool zero = *skip == value;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (long long) gridDim.x * blockDim.x)
        dst[i] = zero ? make_float4(0.f, 0.f, 0.f, 0.f) : const_cast<const float4*>(src)[i];
}
}  // namespace
void copy_or_zero_from_mapped(float* dst, const float* src, long long n, const uint32_t* skip, uint32_t value,
                              void* stream) {
    if (n <= 0) return;
    const long long n4 = n / 4;
    const int blocks = (int) ((n4 + 255) / 256 < 64 ? (n4 + 255) / 256 : 64);
    copy_or_zero_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>((float4*) dst, (const volatile float4*) src, n4, skip,
                                                                    value);
    check("copy_or_zero_from_mapped");
}
// a GPU timestamp (ns, %globaltimer) into buf[i] - the verify window's stage profiler.  Pdl: launched inside a PDL
// stretch, the stamp lets the next kernel launch at once and takes its time when the one before it has finished.
namespace {
template <bool Pdl>
__global__ void gpu_stamp_kernel(unsigned long long* buf, int i) {
    if constexpr (Pdl) {
        pdl_trigger();
        pdl_wait();
    }
    unsigned long long t;
#if defined(STRATA_HIP_GFX906)
    t = wall_clock64() * 40ull;   // gfx906: the wall clock runs at 25 MHz (hipDeviceAttributeWallClockRate) -> ns
#elif defined(__HIPCC__)
    t = wall_clock64() * 10ull;   // gfx10.3 / gfx11 / gfx12: a constant 100 MHz counter, in ns
#else
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
#endif
    buf[i] = t;
}
}  // namespace
void gpu_stamp(unsigned long long* buf, int i, void* stream) {
    if (pdl_scope()) launch_pdl(gpu_stamp_kernel<true>, dim3(1), dim3(1), 0, (cudaStream_t) stream, buf, i);
    else gpu_stamp_kernel<false><<<1, 1, 0, (cudaStream_t) stream>>>(buf, i);
}

// ---- programmatic dependent launch (pdl.hpp)
bool& pdl_scope() {
    static thread_local bool on = false;
    return on;
}

bool pdl_supported() {
#if defined(__HIPCC__) || !defined(CUDART_VERSION) || CUDART_VERSION < 12030
    return false;   // HIP; or a CUDA runtime without the capture-dependency query pdl_launch_ok needs
#else
    // per device (a layer split runs on several): 1 = runs, 2 = does not.  sm_90+ (the card's, or STRATA_EMULATE_CC's),
    // and code built for it: on a build with only older code the driver JIT-compiles PTX without griddepcontrol.
    static std::atomic<int> state[64];
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    int s = state[dev].load(std::memory_order_relaxed);
    if (s == 0) {
        // STRATA_DF_PDL: 1 / 2 = on for every window graph, 0 = off; unset = on for a compute capability 12.0 card (STRATA_SM120_PERF=0 turns
        // that off), where the verifier asks for it for the all-resident graph only (BW-MEASURE, E21: +0.3..+0.5% decode there, bit-identical;
        // -1.6% .. -4.5% in the doorbell graph of a partly resident cache)
        static const int env_mode = [] { const char* v = std::getenv("STRATA_DF_PDL"); return v == nullptr ? -1 : (std::atoi(v) != 0 ? 1 : 0); }();
        int major = 0, minor = 0;
        cudaFuncAttributes fa{};
        const bool on_attr = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
                             cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess;
        const bool env_on = env_mode == 1 || (env_mode < 0 && on_attr && strata::sm120_default(major, minor));
        const bool on = env_on && on_attr &&
                        strata::cc_major_of(major) >= 9 &&
                        cudaFuncGetAttributes(&fa, gpu_stamp_kernel<true>) == cudaSuccess && fa.ptxVersion >= 90 &&
                        fa.binaryVersion >= 90;
        cudaGetLastError();
        s = on ? 1 : 2;
        state[dev].store(s, std::memory_order_relaxed);
    }
    return s == 1;
#endif
}

#if !defined(__HIPCC__)
bool pdl_launch_ok(const void* kernel, cudaStream_t stream) {
#if !defined(CUDART_VERSION) || CUDART_VERSION < 12030
    (void) kernel; (void) stream;
    return false;
#else
    // the kernel's own code is sm_90+ (each kernel checked once per device: its PTX, or a JIT from older PTX, decides)
    {
        static std::mutex mu;
        static std::vector<std::pair<std::pair<const void*, int>, bool>> seen;
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) { cudaGetLastError(); return false; }
        std::lock_guard<std::mutex> lock(mu);
        bool found = false, ok = false;
        for (const auto& e : seen)
            if (e.first.first == kernel && e.first.second == dev) { found = true; ok = e.second; break; }
        if (!found) {
            cudaFuncAttributes fa{};
            ok = cudaFuncGetAttributes(&fa, kernel) == cudaSuccess && fa.ptxVersion >= 90 && fa.binaryVersion >= 90;
            cudaGetLastError();
            seen.push_back({{kernel, dev}, ok});
        }
        if (!ok) return false;
    }
    // every node the next captured node will depend on is a kernel (CUDA allows a programmatic edge only between two
    // kernel nodes): never after a memcpy, memset, host or event node.  A join of branches gives several kernels, a
    // programmatic edge from each; STRATA_DF_PDL=2 allows a single predecessor only.
    static const int mode = [] { const char* v = std::getenv("STRATA_DF_PDL"); return v ? std::atoi(v) : 1; }();
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    const cudaGraphNode_t* deps = nullptr;
    const cudaGraphEdgeData* edges = nullptr;
    size_t n = 0;
#if CUDART_VERSION >= 13000
    const cudaError_t e = cudaStreamGetCaptureInfo(stream, &status, nullptr, nullptr, &deps, &edges, &n);
#else
    const cudaError_t e = cudaStreamGetCaptureInfo_v3(stream, &status, nullptr, nullptr, &deps, &edges, &n);
#endif
    if (e != cudaSuccess) { cudaGetLastError(); return false; }
    if (status != cudaStreamCaptureStatusActive || n == 0 || deps == nullptr) return false;
    if (mode == 2 && n != 1) return false;
    for (size_t i = 0; i < n; ++i) {
        if (edges != nullptr && (edges[i].type != cudaGraphDependencyTypeDefault || edges[i].from_port != 0 ||
                                 edges[i].to_port != 0))
            return false;
        cudaGraphNodeType type;
        if (cudaGraphNodeGetType(deps[i], &type) != cudaSuccess) { cudaGetLastError(); return false; }
        if (type != cudaGraphNodeTypeKernel) return false;
    }
    return true;
#endif
}
#endif

}  // namespace strata::kernels
