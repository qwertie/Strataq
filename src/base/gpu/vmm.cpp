#include "strata/base/gpu/vmm.hpp"

#include <algorithm>

#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)
#include <cuda.h>
#include <cuda_runtime.h>

#include <mutex>

namespace strata::core {

void* find_driver_entry(const char* name) {
    cudaDriverEntryPointQueryResult q{};
    void* p = nullptr;
#if CUDART_VERSION >= 12050
    const cudaError_t e = cudaGetDriverEntryPointByVersion(name, &p, 12000, cudaEnableDefault, &q);
#else   // the versioned query arrived in CUDA 12.5
    const cudaError_t e = cudaGetDriverEntryPoint(name, &p, cudaEnableDefault, &q);
#endif
    if (e != cudaSuccess || q != cudaDriverEntryPointSuccess || p == nullptr) {
        (void) cudaGetLastError();
        return nullptr;
    }
    return p;
}

namespace {

struct Api {
    bool ok = false;
    uint64_t gran = 0;
    int dev = -1;
    decltype(&cuMemAddressReserve) reserve = nullptr;
    decltype(&cuMemAddressFree) free_va = nullptr;
    decltype(&cuMemCreate) create = nullptr;
    decltype(&cuMemRelease) release = nullptr;
    decltype(&cuMemMap) map = nullptr;
    decltype(&cuMemUnmap) unmap = nullptr;
    decltype(&cuMemSetAccess) access = nullptr;
};

// never destroyed: the K/V pools and the cache release their chunks from static destructors at exit
const Api& api() {
    static Api& a = *new Api;
    static std::once_flag once;
    std::call_once(once, [] {
        decltype(&cuDeviceGetAttribute) attr = nullptr;
        decltype(&cuMemGetAllocationGranularity) granularity = nullptr;
        if (cudaGetDevice(&a.dev) != cudaSuccess) return;
        if (!find_driver_entry("cuDeviceGetAttribute", attr) ||
            !find_driver_entry("cuMemGetAllocationGranularity", granularity) ||
            !find_driver_entry("cuMemAddressReserve", a.reserve) || !find_driver_entry("cuMemAddressFree", a.free_va) ||
            !find_driver_entry("cuMemCreate", a.create) || !find_driver_entry("cuMemRelease", a.release) ||
            !find_driver_entry("cuMemMap", a.map) || !find_driver_entry("cuMemUnmap", a.unmap) ||
            !find_driver_entry("cuMemSetAccess", a.access))
            return;
        int vmm = 0;
        if (attr(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, (CUdevice) a.dev) != CUDA_SUCCESS || !vmm)
            return;
        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = a.dev;
        size_t g = 0;
        if (granularity(&g, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS || g == 0) return;
        a.gran = g;
        a.ok = true;
    });
    return a;
}

}  // namespace

bool vmm_available() { return api().ok; }
uint64_t vmm_granularity() { return api().ok ? api().gran : 0; }
uint64_t vmm_min_granularity() { return api().ok ? api().gran : 0; }

VmmChunk vmm_chunk_new(uint64_t bytes) {
    const Api& a = api();
    int dev = 0;
    if (!a.ok || cudaGetDevice(&dev) != cudaSuccess) return 0;
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = dev;
    CUmemGenericAllocationHandle h = 0;
    if (a.create(&h, (size_t) (bytes != 0 ? bytes : a.gran), &prop, 0) != CUDA_SUCCESS) return 0;
    return (VmmChunk) h;
}

void vmm_chunk_free(VmmChunk h) {
    if (h != 0 && api().ok) api().release((CUmemGenericAllocationHandle) h);
}

bool VmmRange::reserve(uint64_t bytes, uint64_t chunk_bytes) {
    release();
    if (base_ != 0)
        return false;
    const Api& a = api();
    chunk_ = chunk_bytes != 0 ? chunk_bytes : a.gran;
    if (!a.ok || bytes == 0 || chunk_ % a.gran != 0 || cudaGetDevice(&dev_) != cudaSuccess) return false;
    const uint64_t n = (bytes + chunk_ - 1) / chunk_;
    CUdeviceptr p = 0;
    if (a.reserve(&p, (size_t) (n * chunk_), 0, 0, 0) != CUDA_SUCCESS) return false;
    base_ = (unsigned long long) p;
    h_.assign((size_t) n, 0);
    return true;
}

void VmmRange::release() {
    if (base_ == 0) return;
    const Api& a = api();
    for (int64_t i = 0; i < chunks(); ++i)
        free_chunk(i);
    if (mapped_count() != 0)
        return;
    a.free_va((CUdeviceptr) base_, (size_t) ((uint64_t) h_.size() * chunk_));
    base_ = 0;
    h_.clear();
}

bool VmmRange::map_at(uint64_t offset, uint64_t bytes, VmmChunk h) {
    return api().map((CUdeviceptr) (base_ + offset), (size_t) bytes, 0, (CUmemGenericAllocationHandle) h, 0) ==
           CUDA_SUCCESS;
}

bool VmmRange::set_access_at(uint64_t offset, uint64_t bytes) {
    CUmemAccessDesc d{};
    d.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    d.location.id = dev_;
    d.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    return api().access((CUdeviceptr) (base_ + offset), (size_t) bytes, &d, 1) == CUDA_SUCCESS;
}

bool VmmRange::unmap_at(uint64_t offset, uint64_t bytes) {
    return api().unmap((CUdeviceptr) (base_ + offset), (size_t) bytes) == CUDA_SUCCESS;
}

}  // namespace strata::core

#else  // not CUDA
// HIP (ROCm 6+): the same elastic range through hipMemAddressReserve / hipMemCreate / hipMemMap / hipMemSetAccess.  Available
// only where the device reports virtual memory management support (STRATA_HIP_VMM=0 turns it off): --kv-grow stays off
// otherwise, as before.  Everything else (gfx906, ROCm older than 6): no virtual memory, the elastic K/V stays off.
#if defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)
#include <hip/hip_runtime.h>
#include <hip/hip_version.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>
#endif

#if defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && defined(HIP_VERSION_MAJOR) && HIP_VERSION_MAJOR >= 6
namespace strata::core {
namespace {
struct Api {
    bool ok = false;
    uint64_t gran = 0, min_gran = 0;
    int dev = -1;
};
void fail(const char* what, hipError_t e) {
    static int n = 0;
    if (n++ < 8) std::fprintf(stderr, "strata: HIP virtual memory: %s failed: %s\n", what, hipGetErrorString(e));
    (void) hipGetLastError();
}
hipMemAllocationProp alloc_prop(int dev) {
    hipMemAllocationProp p{};
    p.type = hipMemAllocationTypePinned;
    p.requestedHandleType = hipMemHandleTypeNone;
    p.location.type = hipMemLocationTypeDevice;
    p.location.id = dev;
    return p;
}
const Api& api() {
    static Api& a = *new Api;   // never destroyed: the K/V pools and the cache release their chunks from static destructors
    static std::once_flag once;
    std::call_once(once, [] {
        const char* off = std::getenv("STRATA_HIP_VMM");
        if (off != nullptr && off[0] == '0') return;
        if (hipGetDevice(&a.dev) != hipSuccess) { (void) hipGetLastError(); return; }
        int vmm = 0;
        if (hipDeviceGetAttribute(&vmm, hipDeviceAttributeVirtualMemoryManagementSupported, a.dev) != hipSuccess || !vmm) {
            (void) hipGetLastError();
            return;
        }
        const hipMemAllocationProp prop = alloc_prop(a.dev);
        size_t g = 0;
        if (hipMemGetAllocationGranularity(&g, &prop, hipMemAllocationGranularityMinimum) != hipSuccess || g == 0) {
            (void) hipGetLastError();
            return;
        }
        // gfx1201 reports a 4 KiB minimum granularity; the elastic range works in 2 MiB chunks like CUDA's (any multiple of the minimum)
        const uint64_t want = 2ull << 20;
        a.min_gran = g;
        a.gran = g >= want ? g : (want + g - 1) / g * g;
        a.ok = true;
    });
    return a;
}
}  // namespace

bool vmm_available() { return api().ok; }
uint64_t vmm_granularity() { return api().ok ? api().gran : 0; }
uint64_t vmm_min_granularity() { return api().ok ? api().min_gran : 0; }

VmmChunk vmm_chunk_new(uint64_t bytes) {
    const Api& a = api();
    int dev = 0;
    if (!a.ok || hipGetDevice(&dev) != hipSuccess) return 0;
    const hipMemAllocationProp prop = alloc_prop(dev);
    hipMemGenericAllocationHandle_t h = nullptr;
    if (const hipError_t e = hipMemCreate(&h, (size_t) (bytes != 0 ? bytes : a.gran), &prop, 0); e != hipSuccess) {
        fail("hipMemCreate", e);
        return 0;
    }
    return (VmmChunk) (uintptr_t) h;
}

void vmm_chunk_free(VmmChunk h) {
    if (h != 0 && api().ok) (void) hipMemRelease((hipMemGenericAllocationHandle_t) (uintptr_t) h);
}

bool VmmRange::reserve(uint64_t bytes, uint64_t chunk_bytes) {
    release();
    if (base_ != 0)
        return false;
    const Api& a = api();
    chunk_ = chunk_bytes != 0 ? chunk_bytes : a.gran;
    if (!a.ok || bytes == 0 || chunk_ % a.min_gran != 0 || hipGetDevice(&dev_) != hipSuccess) return false;
    const uint64_t n = (bytes + chunk_ - 1) / chunk_;
    void* p = nullptr;
    if (const hipError_t e = hipMemAddressReserve(&p, (size_t) (n * chunk_), 0, nullptr, 0); e != hipSuccess) { fail("hipMemAddressReserve", e); return false; }
    base_ = (unsigned long long) (uintptr_t) p;
    h_.assign((size_t) n, 0);
    return true;
}

void VmmRange::release() {
    if (base_ == 0) return;
    for (int64_t i = 0; i < chunks(); ++i)
        free_chunk(i);
    if (mapped_count() != 0)
        return;
    (void) hipMemAddressFree((void*) (uintptr_t) base_, (size_t) ((uint64_t) h_.size() * chunk_));
    base_ = 0;
    h_.clear();
}

bool VmmRange::map_at(uint64_t offset, uint64_t bytes, VmmChunk h) {
    if (const hipError_t e = hipMemMap((void*) (uintptr_t) (base_ + offset), (size_t) bytes, 0,
                                      (hipMemGenericAllocationHandle_t) (uintptr_t) h, 0);
        e != hipSuccess) {
        fail("hipMemMap", e);
        return false;
    }
    return true;
}

bool VmmRange::set_access_at(uint64_t offset, uint64_t bytes) {
    hipMemAccessDesc d{};
    d.location.type = hipMemLocationTypeDevice;
    d.location.id = dev_;
    d.flags = hipMemAccessFlagsProtReadWrite;
    const hipError_t e = hipMemSetAccess((void*) (uintptr_t) (base_ + offset), (size_t) bytes, &d, 1);
    const bool ok = e == hipSuccess;
    if (!ok) fail("hipMemSetAccess", e);
    return ok;
}

bool VmmRange::unmap_at(uint64_t offset, uint64_t bytes) {
    if (hipMemUnmap((void*) (uintptr_t) (base_ + offset), (size_t) bytes) != hipSuccess) {
        (void) hipGetLastError();
        return false;
    }
    return true;
}
}  // namespace strata::core
#else


namespace strata::core {
bool vmm_available() { return false; }
uint64_t vmm_granularity() { return 0; }
uint64_t vmm_min_granularity() { return 0; }
VmmChunk vmm_chunk_new(uint64_t) { return 0; }
void vmm_chunk_free(VmmChunk) {}
bool VmmRange::reserve(uint64_t, uint64_t) { return false; }
void VmmRange::release() {}
bool VmmRange::map_at(uint64_t, uint64_t, VmmChunk) { return false; }
bool VmmRange::set_access_at(uint64_t, uint64_t) { return false; }
bool VmmRange::unmap_at(uint64_t, uint64_t) { return false; }
}  // namespace strata::core

#endif
#endif

// Every backend: the chunks, over its map_at, set_access_at and unmap_at.
namespace strata::core {

int64_t VmmRange::mapped_count() const {
    int64_t n = 0;
    for (const VmmChunk h : h_) n += h != 0;
    return n;
}

bool VmmRange::map_one(int64_t i, VmmChunk h) {
    if (!map_at((uint64_t) i * chunk_, chunk_, h))
        return false;
    h_[(size_t) i] = h;
    return true;
}

bool VmmRange::set_access(int64_t lo, int64_t hi) {
    return hi <= lo || set_access_at((uint64_t) lo * chunk_, (uint64_t) (hi - lo) * chunk_);
}

bool VmmRange::commit_run(int64_t lo, int64_t hi) {
    if (set_access(lo, hi)) return true;
    for (int64_t c = lo; c < hi; ++c) vmm_chunk_free(unmap(c));
    return false;
}

VmmChunk VmmRange::unmap(int64_t i) {
    if (!mapped(i) || pieces_.count(i) > 0)
        return 0;
    const VmmChunk h = h_[(size_t) i];
    if (!unmap_at((uint64_t) i * chunk_, chunk_))
        return 0;
    h_[(size_t) i] = 0;
    return h;
}

bool VmmRange::map_pieces(int64_t i, uint64_t piece_bytes, const std::function<bool(uint8_t*, uint64_t)>& on_piece) {
    const uint64_t granularity = vmm_min_granularity();
    if (i < 0 || i >= chunks() || mapped(i) || granularity == 0 || piece_bytes == 0 || piece_bytes % granularity != 0)
        return false;
    auto& pieces = pieces_[i];
    pieces.bytes = piece_bytes;
    try {
        pieces.h.reserve((size_t) (1 + (chunk_ - 1) / piece_bytes));
        for (uint64_t at = 0; at < chunk_;) {
            const uint64_t offset = (uint64_t) i * chunk_ + at, bytes = std::min(piece_bytes, chunk_ - at);
            const VmmChunk h = vmm_chunk_new(bytes);
            if (h == 0 || !map_at(offset, bytes, h)) {
                vmm_chunk_free(h);
                free_chunk(i);
                return false;
            }
            pieces.h.push_back(h);
            h_[(size_t) i] = pieces.h.front();
            if (!set_access_at(offset, bytes) || (on_piece && !on_piece(base() + offset, bytes))) {
                free_chunk(i);
                return false;
            }
            at += bytes;
        }
    } catch (...) {
        free_chunk(i);
        throw;
    }
    if (pieces.h.size() == 1)
        pieces_.erase(i);
    return true;
}

void VmmRange::free_chunk(int64_t i) {
    const auto it = pieces_.find(i);
    if (it == pieces_.end()) {
        vmm_chunk_free(unmap(i));
        return;
    }
    if (free_pieces(i, it->second.bytes, it->second.h)) {
        pieces_.erase(it);
        h_[(size_t) i] = 0;
    }
}

bool VmmRange::free_pieces(int64_t i, uint64_t piece_bytes, std::vector<VmmChunk>& pieces) {
    bool freed = true;
    for (size_t k = 0; k < pieces.size(); ++k) {
        if (pieces[k] == 0)
            continue;
        const uint64_t at = (uint64_t) k * piece_bytes;
        if (unmap_at((uint64_t) i * chunk_ + at, std::min(piece_bytes, chunk_ - at))) {
            vmm_chunk_free(pieces[k]);
            pieces[k] = 0;
        } else {
            // Keep ownership so a later free_chunk or release can retry the driver's failed unmap.
            freed = false;
        }
    }
    return freed;
}

}  // namespace strata::core
