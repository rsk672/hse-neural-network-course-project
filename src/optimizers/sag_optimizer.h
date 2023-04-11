#pragma once
#include "optimizer.h"

namespace NeuralNetworkApp {

static constexpr double default_learning_speed_sag = 0.01;

class SAGOptimizer : public Optimizer {
public:
    SAGOptimizer() = default;
    SAGOptimizer(double speed) : learning_speed_(speed) {
    }

    void Train(std::vector<Layer>& layers, const ErrorBlock& error_block,
               const std::vector<Vector>& train_input, const std::vector<Vector>& train_output,
               double max_error, size_t max_iter_count) const;

private:
    double learning_speed_ = default_learning_speed_sag;
};

}  // namespace NeuralNetworkApp
