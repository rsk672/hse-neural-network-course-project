#pragma once
#include "optimizer.h"

namespace NeuralNetworkApp {

static constexpr double default_learning_speed_adam = 0.001;
static constexpr double default_beta1 = 0.9;
static constexpr double default_beta2 = 0.999;
static constexpr double default_eps = 10e-8;
static constexpr size_t default_batch_size_adam = 2;

class AdamOptimizer : public Optimizer {
public:
    AdamOptimizer() = default;
    AdamOptimizer(double learning_speed, double beta1, double beta2, double eps, size_t batch_size)
        : learning_speed_(learning_speed),
          beta1_(beta1),
          beta2_(beta2),
          eps_(eps),
          batch_size_(batch_size) {
    }

    void Train(std::vector<Layer>& layers, const ErrorBlock& error_block,
               const std::vector<Vector>& train_input, const std::vector<Vector>& train_output,
               double max_error, size_t max_iter_count) const;

private:
    double learning_speed_ = default_learning_speed_adam;
    double beta1_ = default_beta1;
    double beta2_ = default_beta2;
    double eps_ = default_eps;
    size_t batch_size_ = default_batch_size_adam;
};

}  // namespace NeuralNetworkApp
