#pragma once
#include "../layer.h"
#include "../error_block.h"
#include "../global.h"

namespace NeuralNetworkApp {

enum OptimizerType { GD, SGD, SAG, Adam };

class Optimizer {
public:
    virtual void Train(std::vector<Layer>& layers, const ErrorBlock& error_block,
                       const std::vector<Vector>& train_input,
                       const std::vector<Vector>& train_output, double max_error,
                       size_t max_iter_count) const = 0;

    double GetAverageError(std::vector<Layer>& layers, const ErrorBlock& error_block,
                           const std::vector<Vector>& train_input,
                           const std::vector<Vector>& train_output) const;

    virtual ~Optimizer() {
    }
};

}  // namespace NeuralNetworkApp
