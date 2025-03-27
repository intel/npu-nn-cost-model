#ifndef VPUEM_PIECEWISE_CALC_SUBBLK_SIZE_H
#define VPUEM_PIECEWISE_CALC_SUBBLK_SIZE_H

#include <array>
#include <iostream>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <list>
#include <tuple>
#include "vpu/shave/VPUEM_op_base_dsp.h"

#include "vpu/types.h"
#include "vpu/vpuem_types.h"

namespace VPUNN {
class VPUEMCalcSubblk {
private:
    const bool adaptive_blk_num_en_;
    const int max_blk_num_;
    const int dspArch_;
    const int simd_;

public:
    VPUEMCalcSubblk(bool adaptive_blk_num_en, int max_blk_num,
                    int dspArch)  // dspArch 128 for VPU2.7 and 512 for VPU4.0
            : adaptive_blk_num_en_(adaptive_blk_num_en),
              max_blk_num_(max_blk_num),
              dspArch_(dspArch), 
              simd_(dspArch/8) {
    }

    int round_dim(const int& d0, const int& d1) const {
        if (d0 < d1) {
            return 1;
        } else {
            // proper implementation based on VPUEM, usage of ceil
            return int(std::ceil(static_cast<double>(d0) /d1));
        }
    }

    VPUEM_Subblk_Tensor calc_ts_subblk_size(const int& max_num_data, const int& data_num, const VPUTensor& ts) const {
        // generate the initial blk tensor
        VPUEM_Subblk_Tensor blk_tensor = VPUEM_Subblk_Tensor({1, 1, 1}, ts.get_dtype(), ts.get_layout());

        std::array<unsigned int, 4> reversed_shape;
        std::reverse_copy(ts.get_shape().begin(), ts.get_shape().end(), reversed_shape.begin());

        // compute m by finding first index dim != 1
        int m = 3;
        bool found = false;
        for (long unsigned int idx = 0; idx < reversed_shape.size(); ++idx) {
            if (reversed_shape[idx] != 1) {
                m -= idx;
                found = true;
                break;
            }
        }
        
        if (!found) {
            throw std::runtime_error("All the dimensions are 1");
        } else {
            if (int(ts.get_shape()[m]) >= max_num_data) {
                blk_tensor.set_shape(m - 1, max_num_data);
            } else {
                blk_tensor.set_shape(m - 1, data_num * round_dim(ts.get_shape()[m], data_num));
            }

            if (m >= 2) {
                blk_tensor.set_shape(m - 2, round_dim(max_num_data, blk_tensor.get_shape()[m - 1]));
                if (blk_tensor.get_shape()[m - 2] > int(ts.get_shape()[m - 1])) {
                    blk_tensor.set_shape(m - 2, ts.get_shape()[m - 1]);
                }
                while (blk_tensor.get_shape()[m - 2] * blk_tensor.get_shape()[m - 1] > max_num_data) {
                    if (blk_tensor.get_shape()[m - 2] > 1) {
                        blk_tensor.set_shape(m - 2, blk_tensor.get_shape()[m - 2] - 1);
                    } else {
                        return blk_tensor;
                    }
                }
            }

            if (m == 3) {
                blk_tensor.set_shape(m - 3, round_dim(max_num_data,
                                                      (blk_tensor.get_shape()[m - 1] * blk_tensor.get_shape()[m - 2])));
                if (blk_tensor.get_shape()[m - 3] > int(ts.get_shape()[m - 2])) {
                    blk_tensor.set_shape(m - 3, ts.get_shape()[m - 2]);
                }
                while (blk_tensor.get_shape()[m - 3] * blk_tensor.get_shape()[m - 2] * blk_tensor.get_shape()[m - 1] >
                       max_num_data) {
                    if (blk_tensor.get_shape()[m - 3] > 1) {
                        blk_tensor.set_shape(m - 3, blk_tensor.get_shape()[m - 3] - 1);
                    } else {
                        return blk_tensor;
                    }
                }
            }
            
            return blk_tensor;
        }
    }

    VPUEM_Subblk_Tensor calc_last_subblk_size(const VPUEM_Subblk_Tensor& dim, const VPUTensor& ts) const {
        VPUEM_Subblk_Tensor dim_last =
                VPUEM_Subblk_Tensor({1, 1, dim.get_shape()[2]}, ts.get_dtype(), ts.get_layout());
        dim_last.set_shape(0, int(ts.get_shape()[1]) <= dim.get_shape()[0]
                                      ? dim.get_shape()[0]
                                      : ts.get_shape()[1] % dim.get_shape()[0]);

        dim_last.set_shape(1, int(ts.get_shape()[2]) <= dim.get_shape()[1]
                                      ? dim.get_shape()[1]
                                      : ts.get_shape()[2] % dim.get_shape()[1]);

        if (dim_last.get_shape()[0] == 0 && dim_last.get_shape()[1] == 0) {
            dim_last = dim;
        } else {
            for (int i = 0; i < 2; ++i) {
                if (dim_last.get_shape()[i] == 0) {
                    dim_last.set_shape(i, dim.get_shape()[i]);
                }
            }
        }

        return dim_last;
    }

    std::tuple<std::list<int>, std::list<VPUEM_Subblk_Tensor>, std::list<VPUEM_Subblk_Tensor>,
               std::list<VPUEM_Subblk_Tensor>, std::list<VPUEM_Subblk_Tensor>>
    calc_dsp_block_unit(const std::vector<VPUTensor>& inputs, const std::vector<VPUTensor>& outputs) const {
        std::list<int> numBlocks{1, 1, 1};
        std::list<VPUEM_Subblk_Tensor> isizes, isizes_last, osizes, osizes_last;
        int data_num = 0;
        int max_num_data = simd_;

        VPUEM_OpBaseDSP vpuem_setup = VPUEM_OpBaseDSP(inputs, outputs, adaptive_blk_num_en_, max_blk_num_, dspArch_);
        int max_blk_num = vpuem_setup.get_max_blk_num();
        const auto& input_tensors = inputs;

        for (const VPUTensor& in_tensor : input_tensors) {
            VPUTensor in_ts = VPUTensor(1, 1, 1, in_tensor.volume(), in_tensor.get_dtype(),
                                     in_tensor.get_layout());  // flatten the tensor

            //the correct implementation based on VPUEM
            double precision_factor = dtype_to_bits(in_ts.get_dtype()) / 8.0;
            data_num = int(std::ceil(simd_ / precision_factor));
            max_num_data = data_num * max_blk_num;

            VPUEM_Subblk_Tensor idim = calc_ts_subblk_size(max_num_data, data_num, in_ts);
            isizes.push_back(idim);

            VPUEM_Subblk_Tensor idim_last = calc_last_subblk_size(idim, in_ts);
            isizes_last.push_back(idim_last);
        }

        const auto& output_tensors = outputs;

        for (const VPUTensor& out_tensor : output_tensors) {
            VPUTensor out_ts = VPUTensor(1, 1, 1, out_tensor.volume(), out_tensor.get_dtype(),
                                     out_tensor.get_layout());  // flatten the tensor
            
            // the correct implementation based on VPUEM
            double precision_factor = dtype_to_bits(out_ts.get_dtype()) / 8.0;
            data_num = int(std::ceil(simd_ / precision_factor));
            max_num_data = data_num * max_blk_num;

            VPUEM_Subblk_Tensor odim = calc_ts_subblk_size(max_num_data, data_num, out_ts);
            osizes.push_back(odim);

            VPUEM_Subblk_Tensor odim_last = calc_last_subblk_size(odim, out_ts);
            osizes_last.push_back(odim_last);

            auto it = numBlocks.begin();
            for (int i = 0; i < 3 && it != numBlocks.end(); ++i, ++it) {
                    *it = int(std::ceil(float(out_ts.get_shape()[i + 1]) / float(odim.get_shape()[i])));
            }

            if (out_tensor.get_shape().size() > 4) {
                    auto itt = numBlocks.begin();
                for (long unsigned int i = 0; i < out_tensor.get_shape().size() - 4 && itt != numBlocks.end(); ++i, ++itt) {
                        *itt *= out_tensor.get_shape()[i];
                    }
            }
        }
        return {numBlocks, isizes, isizes_last, osizes, osizes_last};
    }
};

} // namespace VPUNN

#endif // VPUEM_PIECEWISE_CALC_SUBBLK_SIZE_H