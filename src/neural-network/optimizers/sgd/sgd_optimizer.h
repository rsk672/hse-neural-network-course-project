#pragma once
#include "../base_optimizer.h"

namespace NeuralNetworkApp {

class SGDOptimizer : public BaseOptimizer {
public:
    static constexpr double defualt_learning_speed_sgd = 0.1;
    static constexpr double default_batch_size_sgd = 1;

    SGDOptimizer() = default;
    SGDOptimizer(size_t batch_size, double speed)
        : batch_size_(batch_size), learning_speed_(speed) {
    }

    void Train(std::vector<Layer>& layers, const ErrorBlock& error_block,
               const std::vector<Vector>& train_input, const std::vector<Vector>& train_output,
               size_t max_iter_count) const override;

private:
    double learning_speed_ = defualt_learning_speed_sgd;
    size_t batch_size_ = default_batch_size_sgd;

    void InitializePermutation(std::vector<size_t>* perm) const;

    void CalculateGradientsOnBatch(std::vector<Layer>* layers, std::vector<Matrix>* grads_A,
                                   std::vector<Vector>* grads_b, const ErrorBlock& error_block,
                                   const std::vector<size_t>& batch_indices,
                                   const std::vector<Vector>& train_input,
                                   const std::vector<Vector>& train_output) const;

    void UpdateLayerParams(std::vector<Layer>* layers, const std::vector<Matrix>& grads_A,
                           const std::vector<Vector>& grads_b, size_t vectors_count) const;
};

}  // namespace NeuralNetworkApp
