// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_DPU_HALO_H
#define VPUNN_DPU_HALO_H

#include <array>
#include <iostream>
#include <map>
#include <sstream>  //
#include <string>

#include "dpu_types.h"

namespace VPUNN {

/// halo information for a workload
/// halo information: halo values are zero or positive and represent elements that are read from or written to the
/// other tiles. Exception is input halo negative that models more memory tensor versus compute tensor.
/// halo values make sense only when padding is zero. If padding is not zero halo is irrelevant (for inputs).
///
/// Input0 Memory tensor= Input_0 (the compute tensor) minus  input_halo.  If input_halo is negative the memory
/// tensor is larger (VPU40).
///  Output0 Memory Tensor = Output0 plus inbound_halo   (= Output Overlapped tensor , SplitOverlapped for next Layer
///  input)
/// Split Overlapped input (eg SOHO) has input halo=0 by definition!
///
/// SOHK aspects:
/// fist step is arranging the input in SOHO on first 2 tiles identical,  and second 2 tiles identical
/// first T (T0)  will produce half K  (SOK on a quarter) and write them to T0 and T1 (OWT =2), but will write also
/// with HALO (the bottom rows) to Tiles2&3
///  also first T will be written in its output as inbound halo: Half of K from T1, and bottom row ad inbound halo
///  from T2(K/2 ch) and T3(K/2 channels)
class HaloWorkload {
public:
    /// @brief HW 2D dimension attributes of each dimension ends
    struct HaloInfoHW {
    public:
        int top{0};     ///< vertical dim top  or up
        int bottom{0};  ///< vertical dim bottom or down
        int left{0};    ///< horizontal  dim left
        int right{0};   ///< horizontal  dim right

    public:
        HaloInfoHW() = default;
        HaloInfoHW(int top, int bottom, int left, int right): top{top}, bottom{bottom}, left{left}, right{right} {
        }
        bool operator==(const HaloInfoHW& b) const {
            bool r{true};
            r = r && (top == b.top);
            r = r && (bottom == b.bottom);
            r = r && (left == b.left);
            r = r && (right == b.right);
            return r;
        }
        void setVerticalNoHalo() noexcept {
            // H halo must be forced to zero
            top = 0;
            bottom = 0;
        }
    };
    friend inline std::ostream& operator<<(std::ostream& stream, const VPUNN::HaloWorkload::HaloInfoHW& d);

public:
    /// @brief HWC 3 dimension attributes of each dimension ends
    class HaloInfoHWC : public HaloInfoHW {  // if not public will have pybind build problems
    public:
        using HaloInfoHW::bottom;  // exposing
        using HaloInfoHW::left;    // exposing
        using HaloInfoHW::right;   // exposing
        using HaloInfoHW::top;     // exposing

        int front{0};  ///< channel dim ,near, low index channels
        int back{0};   ///< channel dim ,far, high index channels

        using HaloInfoHW::setVerticalNoHalo;

    public:
        HaloInfoHWC() = default;
        HaloInfoHWC(int top, int bottom, int left, int right, int front, int back)
                : HaloInfoHW{top, bottom, left, right}, front{front}, back{back} {
        }
        HaloInfoHWC(int top, int bottom, int left, int right): HaloInfoHW{top, bottom, left, right} {
        }

        bool operator==(const HaloInfoHWC& b) const {
            bool r{true};
            r = r && ((HaloInfoHW)(*this) == (HaloInfoHW)b);  // base part
            r = r && (front == b.front);
            r = r && (back == b.back);
            return r;
        }

        /// this function checks if all the halo data is positive (TBLRFB)
        bool isAllPositive() const {
            if ((top < 0) || (bottom < 0) || (left < 0) || (right < 0) || (back < 0) || (front < 0)) {
                return false;
            }
            return true;
        }
    };

    /// input halo information. The halo represents how many lines or cols (positive value) from the Compute tensor are
    /// not part of memory tensor, but are read from another tile.
    /// This is not present in VPU40 where Overlap is always present (halo is zero),but is active on VPU2.x IDU via ISI
    /// NN trained for VPU40 should ignore these fields, but can be used for memory aspects
    /// For VPU4.0 we can have input (overlapped input compute tensor) that is less than memory tensor (multiple
    /// consumers of previous layer forces maximum halo). This situation is modeled using negative input halo => input
    /// in memory is extended with the negative value.
    ///  affects memory tensor, normally >=0, but also negative accepted for memory extension.
    HaloInfoHWC input_0_halo{};

    /// output halo information. The halo represents how many lines or cols (positive value) from the Compute tensor are
    /// also broadcast to the adjacent tiles, thus increasing the available info for next layer. This is not present in
    /// VPU27, but is core in VPU4.0 via ITI.  NN trained for VPU27 should ignore these
    /// Note: out Tensor + halo does not describe the Next Layer Overlapped input tensors. To obtain the next Layer
    /// Overlapped tensors you need this compute tensors + the halo provided by its adjacent tiles!
    /// next LayerOverlaped tensor is  also our Memory tensor in an extended way. We don't care/use it but other tiles
    /// do.
    /// precondition: value >=0
    HaloInfoHWC output_0_halo{};

    ///@brief to how many tiles the halo will be written (1 means one besides our own tile )
    /// If the output halo is written to more than 1 adjacent tile, for example to 2. A kind of
    /// replicate the halo only, not the full tensor
    /// used by NN. A value makes sense only if output_0_halo is present.
    /// Note: Special: if the full output is written to another tile(s) . use output_write_tiles field to model this.
    /// precondition: value>=0
    HaloInfoHWC output_0_halo_broadcast_cnt{};

    /// @brief Inbound halo, elements written by other tiles in extension of our compute output tensor!
    // these inbound halo are for computation of memory tensor.
    // They do not affect runtime and should not be part of NN descriptor
    HaloInfoHWC output_0_inbound_halo{};

public:
    bool operator==(const HaloWorkload& b) const {
        bool r{true};
        r = r && (input_0_halo == b.input_0_halo);
        r = r && (output_0_halo == b.output_0_halo);
        r = r && (output_0_halo_broadcast_cnt == b.output_0_halo_broadcast_cnt);
        r = r && (output_0_inbound_halo == b.output_0_inbound_halo);
        return r;
    }

    void setVerticalNoHalo() noexcept {
        // H input halo must be forced to zero
        input_0_halo.setVerticalNoHalo();
        // H output halo  must be zero
        output_0_halo.setVerticalNoHalo();
        output_0_halo_broadcast_cnt.setVerticalNoHalo();
        output_0_inbound_halo.setVerticalNoHalo();
    }
    void setInboudHaloVerticalForBradcastAll(const unsigned int full_output_size,
                                             const unsigned int output_remaining_to_process,
                                             const unsigned output_tile_dim) {
        // we assume here that we want broadcast for all tiles

        // if OWT is >1 , it means that we want to broadcast the split to all other tiles
        //   we need to populate output_inbound halo , so that the memory tensor is equal to all output
        //   tensor of the full layer
        //  the top/bottom value depend on the position of the tile in the list.

        const auto prev_tiles_output_processed{full_output_size - output_remaining_to_process};  // sum of prev tiles
        const auto next_tiles_output_to_process{output_remaining_to_process - output_tile_dim};  // sum of next tiles

        output_0_inbound_halo.top = prev_tiles_output_processed;
        output_0_inbound_halo.bottom = next_tiles_output_to_process;
        // memory output tensor:
        // (output_size - remaining_output_to_split) +  (output_tile_dim)
        // +(remaining_output_to_split -output_tile_dim) = output_size , constant for all tiles
    }
};

inline std::ostream& operator<<(std::ostream& stream, const VPUNN::HaloWorkload::HaloInfoHW& d) {
    stream << "[TBLR] :  \t{"                  //
           << d.top << "," << d.bottom << ","  //
           << d.left << "," << d.right << "} ;";
    return stream;
}
inline std::ostream& operator<<(std::ostream& stream, const VPUNN::HaloWorkload::HaloInfoHWC& d) {
    stream << "[TBLRFB] :  \t{"                //
           << d.top << "," << d.bottom << ","  //
           << d.left << "," << d.right << ","  //
           << d.front << "," << d.back << "} ;";
    return stream;
}

inline std::ostream& operator<<(std::ostream& stream, const VPUNN::HaloWorkload& d) {
    stream << "Halo: \n"                                                     //
           << " input " << d.input_0_halo << "\n"                            //
           << " output" << d.output_0_halo << "\n"                           //
           << " output broadcast " << d.output_0_halo_broadcast_cnt << "\n"  //
           << " output inbound" << d.output_0_inbound_halo << "\n"           //
            ;
    return stream;
}

}  // namespace VPUNN

#endif  // VPUNN_TYPES_H
