#pragma once
#include "optimizer.h"

namespace NeuralNetworkApp {

static constexpr double default_learning_speed_gd = 0.1;

class GDOptimizer : public Optimizer {
public:
    GDOptimizer() = default;
    GDOptimizer(double speed) : learning_speed_(speed) {
    }
    void Train(std::vector<Layer>& layers, const ErrorBlock& error_block,
               const std::vector<Vector>& train_input,
               const std::vector<Vector>& train_output, double max_error,
               size_t max_iter_count) const;

private:
    double learning_speed_ = default_learning_speed_gd;
};

}  // namespace NeuralNetworkApp
