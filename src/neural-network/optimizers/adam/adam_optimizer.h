#pragma once
#include "../base_optimizer.h"

namespace NeuralNetworkApp {

class AdamOptimizer : public BaseOptimizer {
public:
    static constexpr double default_learning_speed_adam = 0.001;
    static constexpr double default_beta1 = 0.9;
    static constexpr double default_beta2 = 0.999;
    static constexpr double default_eps = 10e-8;
    static constexpr size_t default_batch_size_adam = 2;

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
               size_t max_iter_count) const override;

private:
    double learning_speed_ = default_learning_speed_adam;
    double beta1_ = default_beta1;
    double beta2_ = default_beta2;
    double eps_ = default_eps;
    size_t batch_size_ = default_batch_size_adam;

    void InitializePermutation(std::vector<size_t>* perm) const;

    void InitializeMomentsBuffers(const std::vector<Layer>& layers,
                                  std::vector<Matrix>* current_m_A,
                                  std::vector<Vector>* current_m_b,
                                  std::vector<Matrix>* current_v_A,
                                  std::vector<Vector>* current_v_b) const;
    void CalculateGradientsOnBatch(std::vector<Layer>* layers, std::vector<Matrix>* grads_A,
                                   std::vector<Vector>* grads_b, const ErrorBlock& error_block,
                                   const std::vector<size_t>& batch_indices,
                                   const std::vector<Vector>& train_input,
                                   const std::vector<Vector>& train_output) const;

    void UpdateMoments(const std::vector<Matrix>& grads_A, const std::vector<Vector>& grads_b,
                       std::vector<Matrix>* current_m_A, std::vector<Vector>* current_m_b,
                       std::vector<Matrix>* current_v_A, std::vector<Vector>* current_v_b) const;

    void UpdateLayerParams(std::vector<Layer>* layers, double current_learning_speed,
                           const std::vector<Matrix>& current_m_A,
                           const std::vector<Vector>& current_m_b,
                           const std::vector<Matrix>& current_v_A,
                           const std::vector<Vector>& current_v_b) const;
};

}  // namespace NeuralNetworkApp
