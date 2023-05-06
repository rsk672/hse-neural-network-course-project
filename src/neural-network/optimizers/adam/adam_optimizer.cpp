#include <limits>
#include <algorithm>
#include <random>
#include <vector>
#include <iostream>
#include "adam_optimizer.h"

namespace NeuralNetworkApp {

void AdamOptimizer::InitializePermutation(std::vector<size_t>* perm) const {
    for (size_t i = 0; i < perm->size(); ++i) {
        perm->at(i) = i;
    }
}

void AdamOptimizer::InitializeMomentsBuffers(const std::vector<Layer>& layers,
                                             std::vector<Matrix>* current_m_A,
                                             std::vector<Vector>* current_m_b,
                                             std::vector<Matrix>* current_v_A,
                                             std::vector<Vector>* current_v_b) const {
    for (size_t j = 0; j < layers.size(); ++j) {
        current_m_A->at(j) = Matrix(layers[j].GetOutputSize(), layers[j].GetInputSize());
        current_m_b->at(j) = Vector(layers[j].GetOutputSize());
        current_v_A->at(j) = Matrix(layers[j].GetOutputSize(), layers[j].GetInputSize());
        current_v_b->at(j) = Vector(layers[j].GetOutputSize());
    }
}

void AdamOptimizer::CalculateGradientsOnBatch(
    std::vector<Layer>* layers, std::vector<Matrix>* grads_A, std::vector<Vector>* grads_b,
    const ErrorBlock& error_block, const std::vector<size_t>& batch_indices,
    const std::vector<Vector>& train_input, const std::vector<Vector>& train_output) const {

    for (size_t j = 0; j < layers->size(); ++j) {
        grads_A->at(j) = Matrix(layers->at(j).GetOutputSize(), layers->at(j).GetInputSize());
        grads_b->at(j) = Vector(layers->at(j).GetOutputSize());
    }

    size_t n = train_input.size();
    for (size_t i = 0; i < std::min(n, batch_size_); ++i) {

        auto x = train_input[batch_indices[i]];
        for (size_t j = 0; j < layers->size(); ++j) {
            x = layers->at(j).PushForward(x);
        }

        Vector u = error_block.GetGradientValue(x, train_output[batch_indices[i]]);

        for (ssize_t j = layers->size() - 1; j >= 0; --j) {
            u = layers->at(j).PushBackwards(u, &grads_A->at(j), &grads_b->at(j));
        }
    }
}

void AdamOptimizer::UpdateMoments(const std::vector<Matrix>& grads_A,
                                  const std::vector<Vector>& grads_b,
                                  std::vector<Matrix>* current_m_A,
                                  std::vector<Vector>* current_m_b,
                                  std::vector<Matrix>* current_v_A,
                                  std::vector<Vector>* current_v_b) const {
    size_t count = current_m_A->size();
    for (size_t i = 0; i < count; ++i) {

        current_m_A->at(i) = beta1_ * current_m_A->at(i) + (1 - beta1_) * grads_A[i];
        current_m_b->at(i) = beta1_ * current_m_b->at(i) + (1 - beta1_) * grads_b[i];

        current_v_A->at(i) = beta2_ * current_v_A->at(i) +
                             (1 - beta2_) * grads_A[i].unaryExpr([](double x) { return x * x; });
        current_v_b->at(i) = beta2_ * current_v_b->at(i) +
                             (1 - beta2_) * grads_b[i].unaryExpr([](double x) { return x * x; });
    }
}

void AdamOptimizer::UpdateLayerParams(std::vector<Layer>* layers, double current_learning_speed,
                                      const std::vector<Matrix>& current_m_A,
                                      const std::vector<Vector>& current_m_b,
                                      const std::vector<Matrix>& current_v_A,
                                      const std::vector<Vector>& current_v_b) const {

    for (size_t i = 0; i < layers->size(); ++i) {
        Matrix A_shift = (-current_learning_speed * current_m_A[i].array() /
                          (current_v_A[i].array().sqrt() - eps_))
                             .matrix();
        Vector b_shift = (-current_learning_speed * current_m_b[i].array() /
                          (current_v_b[i].array().sqrt() - eps_));

        layers->at(i).ShiftParams(A_shift, b_shift);
    }
}

void AdamOptimizer::Train(std::vector<Layer>& layers, const ErrorBlock& error_block,
                          const std::vector<Vector>& train_input,
                          const std::vector<Vector>& train_output, size_t max_iter_count) const {

    std::random_device rd;
    static std::mt19937 mt(rd());

    size_t n = train_input.size();
    std::vector<size_t> perm(n);
    InitializePermutation(&perm);

    size_t layers_count = layers.size();
    std::vector<Matrix> current_m_A(layers_count);
    std::vector<Vector> current_m_b(layers_count);
    std::vector<Matrix> current_v_A(layers_count);
    std::vector<Vector> current_v_b(layers_count);
    InitializeMomentsBuffers(layers, &current_m_A, &current_m_b, &current_v_A, &current_v_b);

    double current_learning_speed = learning_speed_;

    std::cout << "Training: Adam\n";

    size_t iter_count = 0;
    while (iter_count < max_iter_count) {

        std::shuffle(perm.begin(), perm.end(),
                     mt);  // shuffling vector indices to choose random vectors for batch

        std::vector<Matrix> grads_A(layers_count);
        std::vector<Vector> grads_b(layers_count);

        CalculateGradientsOnBatch(&layers, &grads_A, &grads_b, error_block, perm, train_input,
                                  train_output);

        UpdateMoments(grads_A, grads_b, &current_m_A, &current_m_b, &current_v_A, &current_v_b);

        current_learning_speed = learning_speed_ * sqrt(1 - pow(beta2_, iter_count + 1)) /
                                 (1 - pow(beta1_, iter_count + 1));

        UpdateLayerParams(&layers, current_learning_speed, current_m_A, current_m_b, current_v_A,
                          current_v_b);

        ++iter_count;
    }
}
}  // namespace NeuralNetworkApp
