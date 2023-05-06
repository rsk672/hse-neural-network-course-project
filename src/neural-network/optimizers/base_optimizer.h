#pragma once
#include "../layer/layer.h"
#include "../error-block/error_block.h"
#include "global.h"

namespace NeuralNetworkApp {

// enum OptimizerType { GD, SGD, SAG, Adam };

class BaseOptimizer {
public:
    virtual void Train(std::vector<Layer>& layers, const ErrorBlock& error_block,
                       const std::vector<Vector>& train_input,
                       const std::vector<Vector>& train_output, size_t max_iter_count) const = 0;

    virtual ~BaseOptimizer() = default;
};

}  // namespace NeuralNetworkApp
