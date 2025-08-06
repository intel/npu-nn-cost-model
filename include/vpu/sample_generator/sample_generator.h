// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_SAMPLE_GENERATOR_H
#define VPUNN_SAMPLE_GENERATOR_H

#include <random>

#include "vpu/ranges.h"

namespace VPUNN {

/// @brief Generates sample, different distributions, controlled random seed
class Sampler {
private:
    const unsigned int seed_used;    ///< the seed used to initialize the generator
    mutable std::mt19937 generator;  // Standard mersenne_twister_engine seeded with rd()

    static float uniform_prob() {
        return 1.0F;
    }

    static float decreasing_prob(int x) {
        return 1.0F / (x + 1);
    }

public:
    Sampler(): seed_used((std::random_device())()), generator(seed_used) {};
    Sampler(unsigned int forced_seed): seed_used(forced_seed), generator(seed_used) {};
    unsigned int get_seed() const {
        return seed_used;
    };

    float sample_continuous_uniform(float min_interval_closed = 0.0F, float max_interval_open = 1.0F) {
        std::uniform_real_distribution<float> distrib(min_interval_closed, max_interval_open);
        const auto value = distrib(generator);
        return value;
    }

    /**
     * @brief Random sample from the container, uniform distribution
     *
     * @tparam C a container type
     * @param elements a container, must have at least one element
     * @return T a random sample from the container
     */
    template <class C>
    typename C::value_type sample_list(const C& elements) const {
        assert(elements.size() > 0u);
        const int max_choice = static_cast<int>(elements.size()) - 1;
        std::uniform_int_distribution<> distrib(0, max_choice);
        const auto idx = distrib(generator);
        return elements[idx];
    }

    /**
     * @brief Random sample from the container, decreasing distribution,
     * correlates with 1/(x+1)
     * @tparam C a container type
     * @param elements a container, must have at least one element
     * @return T a random sample from the container
     */
    template <class C>
    typename C::value_type sample_list_decrease_prob(const C& elements) const {
        assert(elements.size() > 0u);
        std::vector<double> prob_distributions(elements.size());
        std::generate(prob_distributions.begin(), prob_distributions.end(), [n = 0]() mutable {
            const auto prob = decreasing_prob(n);
            n++;
            return prob;
        });
        std::discrete_distribution<> distrib(prob_distributions.cbegin(), prob_distributions.cend());
        const auto idx = distrib(generator);
        return elements[idx];
    }
};

///  explicit implementation for smart ranges needs to be implemented in a convenient place
template <>
SmartRanges::value_type Sampler::sample_list_decrease_prob<SmartRanges>(const SmartRanges& elements) const;

template <>
MultiSmartRanges::value_type Sampler::sample_list_decrease_prob<MultiSmartRanges>(const MultiSmartRanges& elements) const;

}  // namespace VPUNN

#endif  //
