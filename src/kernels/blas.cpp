// Copyright © 2024 Intel Corporation
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
        // Every iteration I take 8 inputs from A and B do elementwise multiplication and sum
        // After this step I've 4 elements that then I accumulate with the original vector
        const auto base = step * idx;
        auto v1 = _mm_add_ps(_mm_mul_ps(_A[base], _B[base]), _mm_mul_ps(_A[base + 1], _B[base + 1]));
        auto v2 = _mm_add_ps(_mm_mul_ps(_A[base + 2], _B[base + 2]), _mm_mul_ps(_A[base + 3], _B[base + 3]));
        auto v3 = _mm_add_ps(_mm_mul_ps(_A[base + 4], _B[base + 4]), _mm_mul_ps(_A[base + 5], _B[base + 5]));
        auto v4 = _mm_add_ps(_mm_mul_ps(_A[base + 6], _B[base + 6]), _mm_mul_ps(_A[base + 7], _B[base + 7]));
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

// Computes the L2 norm (Euclidean length) of a vector
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
void inline cblas_sgemm_rm_ntt_10(const int M, const int N, const int K, const float* A, const int lda, const float* B,
                                  const int ldb, float* C, const int ldc) {
    for (int i = 0; i < N; ++i) {
        const auto b_idx = i * ldb;
        for (int j = 0; j < M; ++j) {
            const auto a_idx = j * lda;
            C[i + j * ldc] = dot(A, B, K, a_idx, b_idx);
        }
    }
}

// todo: check if compiler can optimize better if helped.
void cblas_sgemm(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M,
                 const int N, const int K, const float alpha, const float* A, const int lda, const float* B,
                 const int ldb, const float beta, float* C, const int ldc) {
    if (layout == CblasRowMajor && TransA == CblasNoTrans && TransB == CblasTrans && alpha == 1 && beta == 0) {
        // Highly optimized GEMM for our specific use case
        cblas_sgemm_rm_ntt_10(M, N, K, A, lda, B, ldb, C, ldc);
        return;
    }
    // naive implementation
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