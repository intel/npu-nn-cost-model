// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_BLAS_H
#define VPUNN_BLAS_H

// Select the CBLAS library using propreccsing macro

#if defined(USE_OPENBLAS) || defined(USE_MKL)
#include <cblas.h>
#else
typedef enum CBLAS_LAYOUT { CblasRowMajor = 101, CblasColMajor = 102 } CBLAS_LAYOUT;
typedef enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 } CBLAS_TRANSPOSE;

// Computes the L2 norm (Euclidian length) of a vector
float cblas_snrm2(const int N, const float* X, const int incX);
// Multiplies each element of a vector by a constant (single-precision).
void cblas_sscal(const int N, const float alpha, float* X, const int incX);
// Perform a Generalized Matrix Matrix operation (single precision)
void cblas_sgemm(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M,
                 const int N, const int K, const float alpha, const float* A, const int lda, const float* B,
                 const int ldb, const float beta, float* C, const int ldc);
#endif

#endif  // VPUNN_BLAS_H
