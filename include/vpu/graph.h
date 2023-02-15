// Copyright © 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_GRAPH_H
#define VPUNN_GRAPH_H

#include <cstddef>
#include <iterator>
#include <memory>
#include <unordered_map>
#include "vpu_layer_cost_model.h"

namespace VPUNN {

/**
 * @brief Base class that represent
 *
 */
class VPUComputeNode {
private:
    std::shared_ptr<DPULayer> dpu;
    std::shared_ptr<SWOperation> shv;
    int _hash;

public:
    /**
     * @brief Node operation type enum
     *
     */
    enum class OpType { DPU_COMPUTE_NODE, SHV_COMPUTE_NODE };

    /**
     * @brief Node type
     *
     */
    OpType type;

    /**
     * @brief Construct a new VPUComputeNode object from a DPU layer
     *
     * @param dpu_op a DPU layer
     */
    VPUComputeNode(const std::shared_ptr<DPULayer> dpu_op): dpu(dpu_op) {
        type = VPUComputeNode::OpType::DPU_COMPUTE_NODE;
        _hash = std::rand();
    }

    /**
     * @brief Construct a new VPUComputeNode object from a SHV layer
     *
     * @param shv_op a SHV layer
     */
    VPUComputeNode(const std::shared_ptr<SWOperation> shv_op): shv(shv_op) {
        type = VPUComputeNode::OpType::SHV_COMPUTE_NODE;
        _hash = std::rand();
    }

    /**
     * @brief Compute the cycles of the VPUComputeNode
     *
     * @param cost_model a reference to a VPULayerCostModel object
     * @return unsigned int execution cycles
     */
    unsigned int cycles(VPULayerCostModel& cost_model, VPULayerStrategy& strategy) const {
        if (type == VPUComputeNode::OpType::SHV_COMPUTE_NODE) {
            return cost_model.Layer(*shv, strategy);
        } else {
            return cost_model.Layer(*dpu, strategy);
        }
    }

    /**
     * @brief Operator == : compare this with rhs
     *
     * @param rhs
     * @return true
     * @return false
     */
    bool operator==(const VPUComputeNode& rhs) const {
        return _hash == rhs._hash;
    }

    /**
     * @brief Generate a has for the node
     *
     * @return size_t
     */
    size_t hash() const {
        return _hash;
    }
};

/**
 * @brief A shared pointer to a compute node
 *
 */
using VPUComputeNodePtr = std::shared_ptr<VPUComputeNode>;

/**
 * @brief An helper class to generate has for VPUComputeNode objects
 *
 */
struct VPUComputeHash {
    /**
     * @brief
     *
     * @param op a VPUComputeNode object
     * @return size_t
     */
    size_t operator()(VPUComputeNodePtr op) const {
        return op->hash();
    }
};

/**
 * @brief A Unordered map of VPUComputeNode
 *
 * @tparam T
 */
template <typename T>
using VPUComputeNodeMap = std::unordered_map<VPUComputeNodePtr, T, VPUComputeHash>;

/**
 * @brief Represent the Computation DAG in a VPU device
 *
 */
class VPUComputationDAG {
private:
    std::list<VPUComputeNodePtr> layers;
    VPUComputeNodeMap<std::vector<VPUComputeNodePtr>> successors;
    VPUComputeNodeMap<std::vector<VPUComputeNodePtr>> predecessors;
    std::list<VPUComputeNodePtr> sources_lst;

public:
    /**
     * @brief Construct a new VPUComputationDAG object
     *
     */
    VPUComputationDAG(){};

    /**
     * @brief Add a node to a VPUComputationDAG
     *
     * @param layer
     * @return VPUComputationDAG&
     */
    VPUComputationDAG& addNode(const VPUComputeNodePtr layer) {
        layers.push_back(layer);
        successors[layer] = {};
        predecessors[layer] = {};
        sources_lst.push_back(layer);
        return *this;
    }

    /**
     * @brief Return true if a VPUComputeNode is in the VPUComputationDAG
     *
     * @param layer layer VPUComputeNode
     * @return true
     * @return false
     */
    bool has(const VPUComputeNodePtr layer) const {
        return std::find(layers.begin(), layers.end(), layer) != layers.end();
    }

    /**
     * @brief add and edge to a VPUComputationDAG
     *
     * @param predecessor the edge source
     * @param successor the edge sink
     * @return VPUComputationDAG&
     */
    VPUComputationDAG& addEdge(const VPUComputeNodePtr source, const VPUComputeNodePtr sink) {
        // If source is not present add it
        if (!has(source)) {
            addNode(source);
        }
        // If sink is not present add it
        if (!has(sink)) {
            addNode(sink);
        }

        // Remove the sink from the list of DAG sources
        sources_lst.remove(sink);

        // Add the nodes to the appropriate adjacency list
        predecessors[sink].push_back(source);
        successors[source].push_back(sink);
        return *this;
    }

    /**
     * @brief Returns the number of nodes
     *
     * @return size_t
     */
    size_t nodes() const {
        return layers.size();
    }

    /**
     * @brief Returns the number of edges
     *
     * @return size_t
     */
    size_t edges() {
        size_t edges = 0;
        for (auto node : layers) {
            edges += successors[node].size();
        }
        return edges;
    }

    /**
     * @brief Returns the list of DAG sources
     *
     * @return std::list<VPUComputeNodePtr>
     */
    std::list<VPUComputeNodePtr> sources() {
        return sources_lst;
    }

    /**
     * @brief Return a reference to layers
     *
     * @return std::list<VPUComputeNode>
     */
    std::list<VPUComputeNodePtr> get_layers() {
        return layers;
    }

    /**
     * @brief Return a list of successors of a layer
     *
     * @param layer a pointer to a VPUComputeNode
     * @return std::vector<VPUComputeNodePtr>
     */
    std::vector<VPUComputeNodePtr> get_successors(const VPUComputeNodePtr layer) {
        return successors[layer];
    }

    /**
     * @brief Return a list of predecessors of a layer
     *
     * @param layer layer a pointer to a VPUComputeNode
     * @return std::vector<VPUComputeNodePtr>
     */
    std::vector<VPUComputeNodePtr> get_predecessors(const VPUComputeNodePtr layer) {
        return predecessors[layer];
    }

    /**
     * @brief A DAG iterator
     *
     */
    struct Iterator {
        /**
         * @brief DAG iterator type
         *
         */
        using iterator_category = std::forward_iterator_tag;
        /**
         * @brief Define pointer arithmetic type
         *
         */
        using difference_type = std::ptrdiff_t;

        /**
         * @brief Construct a new DAG Iterator object
         *
         * @param dag
         * @param all_visited
         */
        Iterator(VPUComputationDAG& dag, bool all_visited = false): dag(dag) {
            // The first node
            current_node_ptr = all_visited ? nullptr : dag.sources_lst.front();
            for (auto& layer : dag.layers) {
                dependendcies[layer] = all_visited ? 0u : static_cast<unsigned int>(dag.predecessors[layer].size());
                visited[layer] = all_visited;
            }
        }

        /**
         * @brief Dereference operator
         *
         * @return VPUComputeNodePtr
         */
        VPUComputeNodePtr operator*() const {
            return current_node_ptr;
        }

        /**
         * @brief Arrow operator
         *
         * @return VPUComputeNodePtr
         */
        VPUComputeNodePtr operator->() {
            return current_node_ptr;
        }

        /**
         * @brief Prefix increment operator
         *
         * @return Iterator&
         */
        Iterator& operator++() {
            if (!current_node_ptr) {
                return *this;
            }
            // Node visited
            visited[current_node_ptr] = true;
            for (auto sinks : dag.successors[current_node_ptr]) {
                // Decrement the dependencies of the successors nodes
                dependendcies[sinks] -= 1;
            }
            for (auto layer : dag.layers) {
                if (dependendcies[layer] == 0 && !visited[layer]) {
                    current_node_ptr = layer;
                    return *this;
                }
            }
            // If no successor if find, set, the current node ptr to nullptr
            current_node_ptr = nullptr;
            return *this;
        }

        /**
         * @brief Postfix increment operator
         *
         * @return Iterator
         */
        Iterator operator++(int) {
            Iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        /**
         * @brief Equality operator between DAG iterators
         *
         * @param a DAG Iterator
         * @param b DAG Iterator
         * @return true
         * @return false
         */
        friend bool operator==(const Iterator& a, const Iterator& b) {
            return a.current_node_ptr == b.current_node_ptr;
        };

        /**
         * @brief Disequality operator between DAG iterators
         *
         * @param a DAG Iterator
         * @param b DAG Iterator
         * @return true
         * @return false
         */
        friend bool operator!=(const Iterator& a, const Iterator& b) {
            return a.current_node_ptr != b.current_node_ptr;
        };

    private:
        VPUComputeNodePtr current_node_ptr;
        VPUComputationDAG& dag;
        VPUComputeNodeMap<unsigned int> dependendcies;
        VPUComputeNodeMap<bool> visited;
    };

    /**
     * @brief DAG Iterator begin
     *
     * @return Iterator
     */
    Iterator begin() {
        return Iterator(*this, false);
    }

    /**
     * @brief DAG Iterator end
     *
     * @return Iterator
     */
    Iterator end() {
        return Iterator(*this, true);
    }
};

}  // namespace VPUNN

#endif  // VPUNN_GRAPH_H