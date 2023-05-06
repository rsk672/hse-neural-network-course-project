#pragma once
#include <iostream>
#include "../base_optimizer.h"

namespace NeuralNetworkApp {

class GDOptimizer : public BaseOptimizer {
public:
    static constexpr double default_learning_speed_gd = 0.1;

    GDOptimizer() = default;
    GDOptimizer(double speed) : learning_speed_(speed) {
    }

    void Train(std::vector<Layer>& layers, const ErrorBlock& error_block,
               const std::vector<Vector>& train_input, const std::vector<Vector>& train_output,
               size_t max_iter_count) const override;

private:
    double learning_speed_ = default_learning_speed_gd;

    void CalculateGradients(std::vector<Layer>* layers, std::vector<Matrix>* grads_A,
                            std::vector<Vector>* grads_b, const ErrorBlock& error_block,
                            const std::vector<Vector>& train_input,
                            const std::vector<Vector>& train_output) const;

    void UpdateLayerParams(std::vector<Layer>* layers, const std::vector<Matrix>& grads_A,
                           const std::vector<Vector>& grads_b, size_t vectors_count) const;
};

}  // namespace NeuralNetworkApp
