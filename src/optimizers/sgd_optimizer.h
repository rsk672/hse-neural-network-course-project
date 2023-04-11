#pragma once
#include "optimizer.h"

namespace NeuralNetworkApp {

static constexpr double defualt_learning_speed_sgd = 0.1;
static constexpr double default_batch_size_sgd = 1;

class SGDOptimizer : public Optimizer {
public:
    SGDOptimizer() = default;
    SGDOptimizer(size_t batch_size, double speed)
        : batch_size_(batch_size), learning_speed_(speed) {
    }
    SGDOptimizer(double speed) : learning_speed_(speed) {
    }
    SGDOptimizer(size_t batch_size) : batch_size_(batch_size) {
    }
    void Train(std::vector<Layer>& layers, const ErrorBlock& error_block,
               const std::vector<Vector>& train_input, const std::vector<Vector>& train_output,
               double max_error, size_t max_iter_count) const;

private:
    double learning_speed_ = defualt_learning_speed_sgd;
    size_t batch_size_ = default_batch_size_sgd;
};

}  // namespace NeuralNetworkApp
