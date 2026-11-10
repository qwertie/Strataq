#pragma once

#include "strata/common/kernels/window_limits.hpp"

#include <cstdint>

namespace strata::kernels {

/// Bounds the token rows accepted by the grouped expert kernels and speculative verifier: one verify / batch window's
/// rows (8; 16 in the CUDA / HIP builds, window_limits.hpp).  A window of up to kVerifyNarrowT rows runs the kernels
/// it always ran.
inline constexpr int kVerifyMaxT = kWindowMaxT;

/// copy_from_mapped, or zeros when *skip == value (n a multiple of 4, 16-byte aligned).
void copy_or_zero_from_mapped(float* dst, const float* src, long long n, const uint32_t* skip, uint32_t value,
                              void* stream);
/// Records the device clock in nanoseconds for a captured window's profiler.
void gpu_stamp(unsigned long long* buf, int i, void* stream);

}  // namespace strata::kernels
