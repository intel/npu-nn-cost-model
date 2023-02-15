// Copyright © 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include <math.h>
#include "kernels/vpunn_blas.h"

#include <stdint.h>
#include <algorithm>  // for std::min
#include <cstring>    // for memset

inline bool is_aligned(const float* p) {
    return !(reinterpret_cast<uintptr_t>(p) % 16);
}

#if defined(__SSE__) && defined(__SSE3__)
#define USE_SIMD
#endif

#ifdef USE_SIMD
#include <immintrin.h>
#endif

inline float dot(const float* A, const float* B, const int N, const int offset_a, const int offset_b) {
// if USE_SIMD is defined I need also to check that the vectors are aligned
#ifdef USE_SIMD
    if (!(is_aligned(A + offset_a) && is_aligned(B + offset_b))) {
#endif
        float result = 0.0;
        for (int k = 0; k < N; k++) {
            // Compute the dot product between A and B
            result += A[k + offset_a] * B[k + offset_b];
        }
        return result;
#ifdef USE_SIMD
    }

    const __m128* _A = reinterpret_cast<const __m128*>(A + offset_a);
    const __m128* _B = reinterpret_cast<const __m128*>(B + offset_b);

    __m128 res_v = _mm_setzero_ps();
    float res = 0;
    constexpr unsigned int step = 8;
    const int N_elements = sizeof(__m128) / sizeof(float) * step;
    const int N_smid_loops = N / N_elements;

    for (int idx = 0; idx < N_smid_loops; idx++) {
        // _mm_add_ps adds two 128-bit vectors of [4 x float], and returns the results of the addition
        // _mm_mul_ps multiplies two 128-bit vectors of [4 x float] and returns the results of the multiplication.
        // Every iteration I take 8 inputs from A and B do elementwise multipilcation and sum
        // After this step I've 4 elements that then I accumulate with the original vector
        auto v1 = _mm_add_ps(_mm_mul_ps(_A[step * idx], _B[step * idx]),
                             _mm_mul_ps(_A[step * idx + 1], _B[step * idx + 1]));
        auto v2 = _mm_add_ps(_mm_mul_ps(_A[step * idx + 2], _B[step * idx + 2]),
                             _mm_mul_ps(_A[step * idx + 3], _B[step * idx + 3]));
        auto v3 = _mm_add_ps(_mm_mul_ps(_A[step * idx + 4], _B[step * idx + 4]),
                             _mm_mul_ps(_A[step * idx + 5], _B[step * idx + 5]));
        auto v4 = _mm_add_ps(_mm_mul_ps(_A[step * idx + 6], _B[step * idx + 6]),
                             _mm_mul_ps(_A[step * idx + 7], _B[step * idx + 7]));
        v1 = _mm_add_ps(v1, v2);
        v3 = _mm_add_ps(v3, v4);
        v1 = _mm_add_ps(v1, v3);
        res_v = _mm_add_ps(res_v, v1);
    }

    // At the end I need to accumulate the result vector
    res_v = _mm_hadd_ps(res_v, res_v);  // sum horizontally
    res_v = _mm_hadd_ps(res_v, res_v);  // (NB: need to do this twice to sum all 4 elements)
    _mm_store_ss(&res, res_v);          // store the result that now is in the 0th element

    // as well as the elements that are left outside the loop
    for (int idx = 0; idx < N % N_elements; idx++) {
        res += A[idx + offset_a + N_smid_loops * N_elements] * B[idx + offset_b + N_smid_loops * N_elements];
    }
    return res;
#endif  // #ifdef USE_SIMD
}

// Computes the L2 norm (Euclidian length) of a vector
float cblas_snrm2(const int N, const float* X, const int incX) {
    float norm = 0;
    if (incX == 1) {
        norm = dot(X, X, N, 0, 0);
    } else {
        for (auto idx = 0; idx < N; idx++) {
            norm += powf(X[idx * incX], 2);
        }
    }
    return sqrtf(norm);
}

// Multiplies each element of a vector by a constant (single-precision).
void cblas_sscal(const int N, const float alpha, float* X, const int incX) {
    for (auto idx = 0; idx < N; idx++) {
        *(X + idx * incX) *= alpha;
    }
}

// The version that needs optimization is the one with CblasRowMajor, CblasNoTrans, CblasTrans, alpha==1 and beta==0
void cblas_sgemm_rm_ntt_10(const int M, const int N, const int K, const float* A, const int lda, const float* B,
                           const int ldb, float* C, const int ldc) {
    int i, j, a_idx, b_idx;

    for (i = 0; i < N; i++) {
        b_idx = i * ldb;
        for (j = 0; j < M; j++) {
            a_idx = j * lda;
            C[i + j * ldc] = dot(A, B, K, a_idx, b_idx);
        }
    }
}

static inline void block_copy_in(float* out, const float* in, int ldout, int ldin, unsigned h, unsigned w,
                                 unsigned h_start, unsigned w_start) {
    for (unsigned y = 0; y < h; y++) {
        for (unsigned x = 0; x < w; x++) {
            out[x + (y * ldout)] = in[x + w_start + ((y + h_start) * ldin)];
        }
    }
}

static inline void block_copy_out(float* out, const float* in, int ldout, int ldin, unsigned h, unsigned w,
                                  unsigned h_start, unsigned w_start) {
    for (unsigned y = 0; y < h; y++) {
        for (unsigned x = 0; x < w; x++) {
            out[x + w_start + ((y + h_start) * ldout)] = in[x + (y * ldin)];
        }
    }
}

// The version that needs optimization is the one with CblasRowMajor, CblasNoTrans, CblasTrans, alpha==1 and beta==0
void cblas_sgemm_rm_ntt_10_cache_blocked(const int M, const int N, const int K, const float* A, const int lda,
                                         const float* B, const int ldb, float* C, const int ldc) {
    // 16 * 8 * 8 floats is 1024 floats which is 4KB per block
    // These parameters are extremely magic, and the L2 cache size really matters -- if it changes, these almost
    // certainly need adjusting. We list them here in order from outer to inner in the loop
    const int N_cache_block_size = std::min(4, N), K_cache_block_size = std::min(1, K),
              M_cache_block_size = std::min(4, M);

    // Register blocking is a bit more complex depending on what instruction set we want to use
    // These need to evenly divide their cache block sizes because we do not check in the loop
    // We list them here in order from outer to inner in the loop
    const int M_register_block_size = std::min(4, M_cache_block_size),
              N_register_block_size = std::min(4, N_cache_block_size);

    // The basic idea here is to pack the data we need into cache, rather than going over the matrices in memory

    float* C_block = (float*)malloc(sizeof(float) * M_cache_block_size * N_cache_block_size);
    float* B_block = (float*)malloc(sizeof(float) * K_cache_block_size * N_cache_block_size);
    float* A_block = (float*)malloc(sizeof(float) * M_cache_block_size * K_cache_block_size);

    memset(C_block, 0, M_cache_block_size * N_cache_block_size * sizeof(float));

    const int M_bound = (M % M_cache_block_size) ? (M - (M % M_cache_block_size)) : M;
    const int N_bound = (N % N_cache_block_size) ? (N - (N % N_cache_block_size)) : N;
    const int K_bound = (K % K_cache_block_size) ? (K - (K % K_cache_block_size)) : K;

    for (int n_c = 0; n_c < N_bound; n_c += N_cache_block_size) {
        for (int k_c = 0; k_c < K_bound; k_c += K_cache_block_size) {
            // Copy this slice of matrix B into the cache block
            block_copy_in(B_block, B, K_cache_block_size, ldb, K_cache_block_size, N_cache_block_size, k_c, n_c);
            for (int m_c = 0; m_c < M_bound; m_c += M_cache_block_size) {
                // Copy this slice of matrix A into the cache block
                block_copy_in(A_block, A, M_cache_block_size, lda, M_cache_block_size, K_cache_block_size, m_c, k_c);
                for (int n_r = 0; n_r < N_cache_block_size; n_r += N_register_block_size) {
                    for (int m_r = 0; m_r < M_cache_block_size; m_r += M_register_block_size) {
                        for (int k_r = 0; k_r < K_cache_block_size; k_r += 1) {
                            // Here we do a M_register_block_size x N_register_block_size recursive matmul
                            // Just written out here as another loop nest

                            for (int y = 0; y < M_register_block_size; y += 1) {
                                for (int x = 0; x < N_register_block_size; x += 1) {
                                    // In here we would use SIMD operations to do the arithmetic
                                    // But actually the compiler will do a decent job of auto-vectorizing this loop!
                                    C_block[x + (y * M_cache_block_size)] += A_block[k_r + (y * M_cache_block_size)] *
                                                                             B_block[x + (k_r * K_cache_block_size)];
                                }
                            }
                        }
                    }
                }

                block_copy_out(C, C_block, ldc, M_cache_block_size, M_cache_block_size, N_cache_block_size, m_c, n_c);
                memset(C_block, 0, M_cache_block_size * N_cache_block_size * sizeof(float));
            }
        }
    }

    // While sorting out the N and M ragged iterations, we have to factor in the K ragged iterations because K is the
    // accumulation dimension We need to make sure that we don't try to fix ragged K iterations twice, so we still skip
    // them in the N and M handling If we have ragged N iterations, that means that the rightmost columns of matrix B
    // still need to be processed, updating the rightmost columns of output
    //~ if (N_bound != N) {
    //~ int k_start = 0;
    //~ if (K_bound != K) { k_start = K_bound; }
    //~ for (int n = N_bound; n < N; n += 1) {
    //~ for (int k = k_start; k < K_bound; k += 1) {
    //~ for (int m = 0; m < M; m += 1) {
    //~ C[n + (m * ldc)] += A[k + (m * lda)] * B[n + (k * ldb)];
    //~ }
    //~ }
    //~ }
    //~ }

    // If we have ragged M iterations, that means that the bottom rows of matrix A still need to be processed, updating
    // the bottom rows of output
    //~ if (M_bound != M) {
    //~ int k_start = 0;
    //~ if (K_bound != K) { k_start = K_bound; }
    //~ for (int n = 0; n < N; n += 1) {
    //~ for (int k = k_start; k < K_bound; k += 1) {
    //~ for (int m = M_bound; m < M; m += 1) {
    //~ C[n + (m * ldc)] += A[k + (m * lda)] * B[n + (k * ldb)];
    //~ }
    //~ }
    //~ }
    //~ }

    // We've sorted out the ragged N and M iterations by this point
    // If we have ragged K iterations, that means that *every* point in the output needs to have additional elements
    // accumulated in
    //~ if (K_bound != K) {
    //~ for (int n = 0; n < N; n += 1) {
    //~ for (int k = K_bound; k < K; k += 1) {
    //~ for (int m = 0; m < M; m += 1) {
    //~ C[n + (m * ldc)] += A[k + (m * lda)] * B[n + (k * ldb)];
    //~ }
    //~ }
    //~ }
    //~ }

    //~ free(C_block);
    //~ free(B_block);
    //~ free(A_block);
}

void cblas_sgemm(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M,
                 const int N, const int K, const float alpha, const float* A, const int lda, const float* B,
                 const int ldb, const float beta, float* C, const int ldc) {
    if (layout == CblasRowMajor && TransA == CblasNoTrans && TransB == CblasTrans && alpha == 1 && beta == 0) {
        // Highly optimized GEMM for our specific usecase
        cblas_sgemm_rm_ntt_10(M, N, K, A, lda, B, ldb, C, ldc);
        //~ cblas_sgemm_rm_ntt_10_cache_blocked(M, N, K, A, lda, B, ldb, C, ldc);
        return;
    }
    // naive implemtation
    // Column major
    const CBLAS_TRANSPOSE _TransA = layout != CblasRowMajor ? TransA : TransB;
    const CBLAS_TRANSPOSE _TransB = layout != CblasRowMajor ? TransB : TransA;
    const int _N = layout != CblasRowMajor ? N : M;
    const int _M = layout != CblasRowMajor ? M : N;
    const int _lda = layout != CblasRowMajor ? lda : ldb;
    const int _ldb = layout != CblasRowMajor ? ldb : lda;
    const float* _A = layout != CblasRowMajor ? A : B;
    const float* _B = layout != CblasRowMajor ? B : A;

    long int i, j, k, a_idx, b_idx;
    float result = 0.0;

    for (i = 0; i < _M; i++) {
        for (j = 0; j < _N; j++) {
            result = 0.0;
            for (k = 0; k < K; k++) {
                // Compute a and b idx
                a_idx = _TransA == CblasNoTrans ? i + k * _lda : i * _lda + k;
                b_idx = _TransB == CblasNoTrans ? k + j * _ldb : k * _ldb + j;
                result += _A[a_idx] * _B[b_idx];
            }
            C[i + j * ldc] = C[i + j * ldc] * beta + alpha * result;
        }
    }
    return;
}