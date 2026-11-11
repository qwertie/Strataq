// include/strata/base/gpu/runtime.hpp - the device runtime as shared code sees it: memory, copies and errors, without
// the CUDA runtime's own names, so that code above this layer builds for every backend.  Streams are passed as `void*`
// (a cudaStream_t; null: the legacy default stream).  Each call works on the calling thread's current device.
#pragma once

#include <cstddef>
#include <string>

namespace strata::core {

/// Allocates `bytes` of device memory and zeroes it; nullptr when the device has not that much.
void* gpu_alloc_zeroed(size_t bytes);
/// Allocates uninitialized device memory; nullptr when the runtime refuses it.
void* gpu_alloc(size_t bytes);
void gpu_free(void* p);
/// Queues a copy of `height` rows of `width` bytes, whose starts are `src_pitch` and `dst_pitch` bytes apart, between
/// any two memories (the runtime tells host from device by the addresses).  Returns false when the runtime refuses it.
bool gpu_copy_2d_async(void* dst, size_t dst_pitch, const void* src, size_t src_pitch, size_t width, size_t height,
                       void* stream);
bool gpu_memset_async(void* p, int value, size_t bytes, void* stream);
/// Waits for queued work on stream; false when the runtime reports a failure.
bool gpu_stream_synchronize(void* stream);
/// Copies `bytes` between any two memories and waits for the copy; false when the runtime refuses it.
bool gpu_copy(void* dst, const void* src, size_t bytes);
/// Allocates `bytes` of page-locked host memory that the current device reads in place: `device_view` gets the
/// device's address of it.  nullptr when the runtime cannot lock that much.
void* gpu_host_alloc_mapped(size_t bytes, void** device_view);
/// Frees memory that gpu_host_alloc_mapped gave.
void gpu_host_free(void* p);

/// Tells whether a GPU runtime error's text (cudaGetErrorString's, or a message that quotes it) says the device context
/// is gone: after an illegal memory access, a launch failure or a hardware exception every later call of the process
/// fails too.  Read by the text, so that CUDA, HIP and the compat runtimes are read the same way.
inline bool is_context_lost_error(const std::string& text) {
    static const char* const kLost[] = {"illegal memory access", "illegal address", "launch failure",
                                        "hardware exception", "hardware stack error", "illegal instruction",
                                        "misaligned address", "invalid program counter", "uncorrectable",
                                        "launch timed out", "device-side assert", "device lost", "ERROR_DEVICE_LOST"};
    for (const char* lost : kLost)
        if (text.find(lost) != std::string::npos)
            return true;
    return false;
}
/// Gets the runtime's sticky error text when is_context_lost_error holds for it, else "".
std::string get_lost_context_error();

}  // namespace strata::core
