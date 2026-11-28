#pragma once
// The decode steps' routing, counted on the GPU for the Monitor's experts/token and PRES (the paged engine's STATS
// line), process-wide like strata/base/gpu/traffic.hpp's counts.  The totals live in host-mapped memory, so the GPU
// adds to them and the CPU reads them without a copy or a wait.  There is one per layer: a layer's passes run in
// order on one stream, so no two writers meet.

#include "strata/base/gpu/runtime.hpp"
#include <cstdint>
#include <cstring>
#include <new>

namespace strata::kernels {

/// Totals of routing: each row's trips to experts, and the distinct experts each pass of rows reached; both count
/// the shared experts too.
struct RoutingCounts {
    int64_t routings = 0, experts = 0;
};

namespace routing_detail {
inline RoutingCounts* host = nullptr;     ///< per layer, then the int64 switch of set_routing_counting
inline RoutingCounts* device = nullptr;   ///< host as the GPU addresses it
inline int layers = 0, experts = 0, shared = 0;
}  // namespace routing_detail

/// Allocates the totals of `n_layers` layers of `n_experts` routed experts, each row also running `n_shared`; a
/// paged model calls it once, before its first decode step.
inline void init_routing_counts(int n_layers, int n_experts, int n_shared) {
    using namespace routing_detail;
    const size_t bytes = sizeof(RoutingCounts) * n_layers + sizeof(int64_t);
    void* view = nullptr;
    host = static_cast<RoutingCounts*>(strata::core::gpu_host_alloc_mapped(bytes, &view));
    if (!host)
        throw std::bad_alloc();
    std::memset(host, 0, bytes);
    device = static_cast<RoutingCounts*>(view);
    layers = n_layers;
    experts = n_experts;
    shared = n_shared;
}

/// Turns counting on during decode steps and off otherwise (DeepSeek reads short prompts through the same
/// routing).  Kernels already queued see the change when they run.
inline void set_routing_counting(bool on) {
    if (routing_detail::host)
        *reinterpret_cast<int64_t*>(routing_detail::host + routing_detail::layers) = on;
}

/// Gets the totals so far; callable from any thread at any time (a pass may be half added).
inline RoutingCounts get_routing_counts() {
    RoutingCounts sum;
    for (int l = 0; l < routing_detail::layers; ++l) {
        sum.routings += routing_detail::host[l].routings;
        sum.experts += routing_detail::host[l].experts;
    }
    return sum;
}

/// Queues the count of `layer`'s picks on `stream`, `rows` rows of `k` expert ids `stride` ids apart; nothing
/// before init_routing_counts.
void count_routing(const int32_t* ids, int rows, int k, int stride, int layer, void* stream);

}  // namespace strata::kernels
