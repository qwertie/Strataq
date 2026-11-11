// lost_context_test: which runtime errors is_context_lost_error (include/strata/base/gpu/runtime.hpp) reads as a lost
// GPU context, by the runtime's own texts (cudaGetErrorString needs no GPU).  The engines end on those (#4350).
#include "strata/base/gpu/runtime.hpp"

#include <cuda_runtime.h>

#include <cstdio>

int main() {
    struct Case {
        int code;
        bool lost;
    };
    // codes that CUDA's documentation says leave the process unusable, and some that do not (HIP's share the numbers)
    static const Case kCases[] = {
        {214, true},   // cudaErrorECCUncorrectable, hipErrorECCNotCorrectable
        {700, true},   // cudaErrorIllegalAddress, hipErrorIllegalAddress
        {702, true},   // cudaErrorLaunchTimeout, hipErrorLaunchTimeOut
        {710, true},   // cudaErrorAssert, hipErrorAssert
        {719, true},   // cudaErrorLaunchFailure, hipErrorLaunchFailure
#if !defined(STRATA_USE_HIP)   // (codes HIP does not have)
        {714, true},   // cudaErrorHardwareStackError
        {715, true},   // cudaErrorIllegalInstruction
        {716, true},   // cudaErrorMisalignedAddress
        {718, true},   // cudaErrorInvalidPc
#endif
        {1, false},    // cudaErrorInvalidValue
        {2, false},    // cudaErrorMemoryAllocation, hipErrorOutOfMemory
        {101, false},  // cudaErrorInvalidDevice, hipErrorInvalidDevice
        {600, false},  // cudaErrorNotReady, hipErrorNotReady
    };
    int failures = 0;
    for (const Case& c : kCases) {
        const char* text = cudaGetErrorString((cudaError_t) c.code);
        const bool lost = strata::core::is_context_lost_error(text);
        std::printf("%-5s %d \"%s\"\n", lost == c.lost ? "ok" : "FAIL", c.code, text);
        failures += lost != c.lost;
    }
    // an engine's message quotes the runtime's text after its own words
    failures +=
        !strata::core::is_context_lost_error("cudaStreamSynchronize(s): an illegal memory access was encountered");
    failures += strata::core::is_context_lost_error("the conversation's memory is full: no free block");
    std::printf("lost_context_test: %d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
