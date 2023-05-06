#pragma once
#include "../base_optimizer.h"

namespace NeuralNetworkApp {

class SAGOptimizer : public BaseOptimizer {
public:
    static constexpr double default_learning_speed_sag = 0.01;

    SAGOptimizer() = default;
    SAGOptimizer(double speed) : learning_speed_(speed) {
    }

    void Train(std::vector<Layer>& layers, const ErrorBlock& error_block,
               const std::vector<Vector>& train_input, const std::vector<Vector>& train_output,
               size_t max_iter_count) const override;

private:
    double learning_speed_ = default_learning_speed_sag;

    void InitializeStoredGradientBuffers(const std::vector<Layer>& layers,
                                         std::vector<std::vector<Matrix>>* grads_A_all,
                                         std::vector<std::vector<Vector>>* grads_b_all) const;

    void InitializeGradientSumBuffers(const std::vector<Layer>& layers,
                                      std::vector<Matrix>* sum_grads_A,
                                      std::vector<Vector>* sum_grads_b) const;
    void CalculateGradientOnVector(std::vector<Layer>* layers, std::vector<Matrix>* grads_A,
                                   std::vector<Vector>* grads_b, const ErrorBlock& error_block,
                                   const std::vector<Vector>& train_input,
                                   const std::vector<Vector>& train_output,
                                   size_t vector_index) const;

    void UpdateGradientSum(std::vector<Matrix>* sum_grads_A, std::vector<Vector>* sum_grads_b,
                           const std::vector<std::vector<Matrix>>& grads_A_all,
                           const std::vector<std::vector<Vector>>& grads_b_all,
                           const std::vector<Matrix>& grads_A, const std::vector<Vector>& grads_b,
                           size_t random_index) const;

    void UpdateLayerParams(std::vector<Layer>* layers, const std::vector<Matrix>& sum_grads_A,
                           const std::vector<Vector>& sum_grads_b, size_t encountered_count) const;
};

}  // namespace NeuralNetworkApp
