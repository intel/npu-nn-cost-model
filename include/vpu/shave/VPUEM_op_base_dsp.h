// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_VPUEM_OPBASEDSP_H
#define VPUNN_VPUEM_OPBASEDSP_H

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

#include "vpu/types.h"

namespace VPUNN {
class VPUEM_OpBaseDSP {
private:
    std::vector<VPUTensor> inputs_;
    std::vector<VPUTensor> outputs_;
    bool adaptive_blk_num_en_;
    int max_blk_num_;
    int dspArch_;

public:
    VPUEM_OpBaseDSP(std::vector<VPUTensor> inputs, std::vector<VPUTensor> outputs,
                    bool adaptive_blk_num_en = true, int max_blk_num = 32, int dspArch = 128) // dspArch 128 for VPU2.7 and 512 for VPU4.0
            : inputs_(std::move(inputs)),
              outputs_(std::move(outputs)),
              adaptive_blk_num_en_(adaptive_blk_num_en),
              max_blk_num_(max_blk_num),
              dspArch_(dspArch) {
    }

    void setup_dsp() {
        if (adaptive_blk_num_en_) {
            float size_simd_r = 0.0f;
            int total_io_size = 0;

            for (const VPUTensor& in_tensor : inputs_) {
                total_io_size += in_tensor.size();
            }

            for (const VPUTensor& out_tensor : outputs_) {
                total_io_size += out_tensor.size();
            }


            size_simd_r = float(total_io_size) * 8.0f / float(dspArch_);


            if (size_simd_r <= 1024) {
                max_blk_num_ = 8;
            } else if (size_simd_r <= 8 * 1024) {
                max_blk_num_ = 16;
            } else if (size_simd_r <= 16 * 1024) {
                max_blk_num_ = 32;
            } else if (size_simd_r <= 32 * 1024) {
                max_blk_num_ = 64;
            } else if (size_simd_r <= 64 * 1024) {
                max_blk_num_ = 128;
            } else {
                max_blk_num_ = 32;
            }

            max_blk_num_ *= (dspArch_ == 128 ? 2 : 1);
            max_blk_num_ = std::min(max_blk_num_, 32);
        }
    }

    int get_max_blk_num() {
        setup_dsp();
        return max_blk_num_;
    }
};
}  // namespace VPUNN

#endif // VPUNN_VPUEM_OPBASEDSP_H
