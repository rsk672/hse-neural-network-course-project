#include "sgd_momentum_optimizer.h"
#include <limits>
#include <algorithm>
#include <random>
#include <vector>
#include <iostream>

namespace NeuralNetworkApp {

void SGDMomentumOptimizer::InitializePermutation(std::vector<size_t>* perm) const {
    for (size_t i = 0; i < perm->size(); ++i) {
        perm->at(i) = i;
    }
}

void SGDMomentumOptimizer::InitializeInertionBuffers(const std::vector<Layer>& layers,
                                                     std::vector<Matrix>* inertions_A,
                                                     std::vector<Vector>* inertions_b) const {
    for (size_t j = 0; j < layers.size(); ++j) {
        inertions_A->at(j) = Matrix(layers[j].GetOutputSize(), layers[j].GetInputSize());
        inertions_b->at(j) = Vector(layers[j].GetOutputSize());
    }
}

void SGDMomentumOptimizer::CalculateGradientsOnBatch(
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

void SGDMomentumOptimizer::UpdateInertions(const std::vector<Matrix>& grads_A,
                                           const std::vector<Vector>& grads_b,
                                           std::vector<Matrix>* inertions_A,
                                           std::vector<Vector>* inertions_b) const {
    size_t count = grads_A.size();
    for (size_t i = 0; i < count; ++i) {
        inertions_A->at(i) =
            momentum_ * inertions_A->at(i) + learning_speed_ * grads_A[i] / batch_size_;
        inertions_b->at(i) =
            momentum_ * inertions_b->at(i) + learning_speed_ * grads_b[i] / batch_size_;
    }
}

void SGDMomentumOptimizer::UpdateLayerParams(std::vector<Layer>* layers,
                                             const std::vector<Matrix>& inertions_A,
                                             const std::vector<Vector>& inertions_b) const {
    size_t layers_count = layers->size();
    for (size_t i = 0; i < layers_count; ++i) {
        layers->at(i).ShiftParams(-inertions_A[i], -inertions_b[i]);
    }
}

void SGDMomentumOptimizer::Train(std::vector<Layer>& layers, const ErrorBlock& error_block,
                                 const std::vector<Vector>& train_input,
                                 const std::vector<Vector>& train_output,
                                 size_t max_iter_count) const {

    static std::random_device rd;
    static std::mt19937 mt(rd());

    size_t layers_count = layers.size();
    size_t n = train_input.size();
    std::vector<size_t> perm(n);
    InitializePermutation(&perm);

    std::vector<Matrix> inertions_A(layers_count);
    std::vector<Vector> inertions_b(layers_count);
    InitializeInertionBuffers(layers, &inertions_A, &inertions_b);

    std::cout << "Training: Stochastic Gradient Descent With Momentum" << std::endl;
    size_t iter_count = 0;
    while (iter_count < max_iter_count) {
        std::shuffle(perm.begin(), perm.end(), mt);
        std::vector<Matrix> grads_A(layers_count);
        std::vector<Vector> grads_b(layers_count);
        CalculateGradientsOnBatch(&layers, &grads_A, &grads_b, error_block, perm, train_input,
                                  train_output);
        UpdateInertions(grads_A, grads_b, &inertions_A, &inertions_b);
        UpdateLayerParams(&layers, inertions_A, inertions_b);
        ++iter_count;
    }
}
}  // namespace NeuralNetworkApp
