#pragma once
#include "../base_optimizer.h"

namespace NeuralNetworkApp {

class SGDMomentumOptimizer : public BaseOptimizer {
public:
    static constexpr double defualt_learning_speed_sgd = 0.1;
    static constexpr double default_batch_size_sgd = 1;
    static constexpr double default_momentum_sgd = 0.9;

    SGDMomentumOptimizer() = default;
    SGDMomentumOptimizer(size_t batch_size, double speed, double momentum)
        : batch_size_(batch_size), learning_speed_(speed), momentum_(momentum) {
    }

    void Train(std::vector<Layer>& layers, const ErrorBlock& error_block,
               const std::vector<Vector>& train_input, const std::vector<Vector>& train_output,
               size_t max_iter_count) const override;

private:
    double momentum_ = default_momentum_sgd;
    double learning_speed_ = defualt_learning_speed_sgd;
    size_t batch_size_ = default_batch_size_sgd;

    void InitializePermutation(std::vector<size_t>* perm) const;

    void InitializeInertionBuffers(const std::vector<Layer>& layers,
                                   std::vector<Matrix>* inertions_A,
                                   std::vector<Vector>* inertions_b) const;

    void CalculateGradientsOnBatch(std::vector<Layer>* layers, std::vector<Matrix>* grads_A,
                                   std::vector<Vector>* grads_b, const ErrorBlock& error_block,
                                   const std::vector<size_t>& batch_indices,
                                   const std::vector<Vector>& train_input,
                                   const std::vector<Vector>& train_output) const;

    void UpdateInertions(const std::vector<Matrix>& grads_A, const std::vector<Vector>& grads_b,
                         std::vector<Matrix>* inertions_A, std::vector<Vector>* inertions_b) const;

    void UpdateLayerParams(std::vector<Layer>* layers, const std::vector<Matrix>& inertions_A,
                           const std::vector<Vector>& inertions_b) const;
};

}  // namespace NeuralNetworkApp
