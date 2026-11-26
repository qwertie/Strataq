// src/common/kernels/prefill/gemm_device_test.cu - Gemm::f16 called while another GPU is current gives the products
// it gives with its own GPU current, bitwise (GPU, synthetic, no model; 77 with fewer than 2 GPUs).
//
// Regression test: a layer split's prompt path once reached a card's Gemm with the other card current.  cuBLAS then
// answered some shapes (here N 4096, K 2048, T 318) with CUBLAS_STATUS_INTERNAL_ERROR for every algorithm, and the
// engine ended with "prefill gemm: cublasGemmEx f16: cuBLAS status 14".
#include "strata/common/kernels/prefill/gemm.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {
__global__ void fill_f16(__half* p, int64_t n, unsigned seed) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    unsigned x = (unsigned) i * 2654435761u + seed;
    x ^= x >> 16;
    x *= 2246822519u;
    x ^= x >> 13;
    p[i] = __float2half((float) (x & 0xffff) / 32768.0f - 1.0f);
}

bool ok(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        std::printf("FAIL: %s: %s\n", what, cudaGetErrorString(e));
    return e == cudaSuccess;
}
}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) {
        std::printf("gemm_device_test: skipped, it needs 2 GPUs (%d found)\n", count);
        return 77;
    }
    const int64_t N = 4096, K = 2048, t_max = 1024;
    if (!ok(cudaSetDevice(1), "device 1") || !ok(cudaFree(nullptr), "device 1's context") ||
        !ok(cudaSetDevice(0), "device 0"))
        return 1;
    cudaStream_t st = nullptr;
    __half *W = nullptr, *X = nullptr;
    float* Y = nullptr;
    if (!ok(cudaStreamCreate(&st), "stream") || !ok(cudaMalloc(&W, N * K * 2), "W") ||
        !ok(cudaMalloc(&X, t_max * K * 2), "X") || !ok(cudaMalloc(&Y, t_max * N * 4), "Y"))
        return 1;
    fill_f16<<<(unsigned) ((N * K + 255) / 256), 256, 0, st>>>(W, N * K, 1u);
    fill_f16<<<(unsigned) ((t_max * K + 255) / 256), 256, 0, st>>>(X, t_max * K, 2u);
    strata::prefill::Gemm gemm;
    std::string err;
    if (!gemm.init(st, 0, err)) {
        std::printf("FAIL: %s\n", err.c_str());
        return 1;
    }
    const auto* w = reinterpret_cast<const uint16_t*>(W);
    const auto* x = reinterpret_cast<const uint16_t*>(X);
    std::vector<float> own, other;
    int differ = 0;
    for (const int64_t T : {1, 7, 64, 100, 255, 318, 512, 777, 1024}) {
        own.resize((size_t) (T * N));
        other.resize(own.size());
        gemm.f16(x, w, Y, T, N, K);
        if (!ok(cudaMemcpyAsync(own.data(), Y, own.size() * 4, cudaMemcpyDeviceToHost, st), "own result") ||
            !ok(cudaStreamSynchronize(st), "own GEMM") || !ok(cudaSetDevice(1), "device 1"))
            return 1;
        gemm.f16(x, w, Y, T, N, K);   // (ends the test with cuBLAS's status if it fails)
        int current = -1;
        if (!ok(cudaGetDevice(&current), "current device") || !ok(cudaSetDevice(0), "device 0") ||
            !ok(cudaMemcpyAsync(other.data(), Y, other.size() * 4, cudaMemcpyDeviceToHost, st), "other result") ||
            !ok(cudaStreamSynchronize(st), "the other GEMM"))
            return 1;
        if (current != 1) {
            std::printf("FAIL: T %lld: Gemm::f16 left device %d current, not the caller's 1\n", (long long) T, current);
            return 1;
        }
        if (own != other) {
            std::printf("FAIL: T %lld: the products differ\n", (long long) T);
            ++differ;
        }
    }
    if (differ > 0)
        return 1;
    std::printf("gemm_device_test: every product the same with the other GPU current\n");
    return 0;
}
