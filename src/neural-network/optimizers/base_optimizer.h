#pragma once
#include "../layer/layer.h"
#include "../error-block/error_block.h"
#include "global.h"
#include <iostream>

namespace NeuralNetworkApp {

// enum OptimizerType { GD, SGD, SAG, Adam };

class BaseOptimizer {
public:
    virtual void Train(std::vector<Layer>& layers, const ErrorBlock& error_block,
                       const std::vector<Vector>& train_input,
                       const std::vector<Vector>& train_output, size_t max_iter_count) const = 0;

    // double GetAverageError(std::vector<Layer>& layers, const ErrorBlock& error_block,
    //                        const std::vector<Vector>& train_input,
    //                        const std::vector<Vector>& train_output) const {
    //     size_t layers_count = layers.size();
    //     size_t n = train_input.size();
    //     double train_error = 0;
    //     for (size_t i = 0; i < n; ++i) {
    //         auto x = train_input[i];
    //         for (size_t j = 0; j < layers_count; ++j) {
    //             x = layers[j].PushForward(x);
    //         }
    //         train_error += error_block.GetErrorValue(x, train_output[i]);
    //     }

    //     std::cout << "Train error: " << train_error << "\n";

    //     return train_error / n;
    // }

    virtual ~BaseOptimizer() = default;
};

}  // namespace NeuralNetworkApp
