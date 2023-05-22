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

void SGDMomentumOptimizer::InitializeParamsBuffers(const std::vector<Layer>& layers,
                                                   std::vector<Matrix>* prev_A,
                                                   std::vector<Vector>* prev_b,
                                                   std::vector<Matrix>* current_A,
                                                   std::vector<Vector>* current_b) const {
    for (size_t j = 0; j < layers.size(); ++j) {
        prev_A->at(j) = Matrix(layers[j].GetOutputSize(), layers[j].GetInputSize());
        prev_b->at(j) = Vector(layers[j].GetOutputSize());
        current_A->at(j) = Matrix(layers[j].GetOutputSize(), layers[j].GetInputSize());
        current_b->at(j) = Vector(layers[j].GetOutputSize());
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

void SGDMomentumOptimizer::UpdateLayerParams(
    std::vector<Layer>* layers, const std::vector<Matrix>& grads_A,
    const std::vector<Vector>& grads_b, std::vector<Matrix>* prev_A, std::vector<Vector>* prev_b,
    std::vector<Matrix>* current_A, std::vector<Vector>* current_b, size_t vectors_count) const {
    size_t layers_count = layers->size();
    for (size_t j = 0; j < layers_count; ++j) {
        layers->at(j).ShiftParams(
            -grads_A[j] * learning_speed_ / vectors_count +
                momentum_ / vectors_count * (current_A->at(j) - prev_A->at(j)),
            -grads_b[j] * learning_speed_ / vectors_count +
                momentum_ / vectors_count * (current_b->at(j) - prev_b->at(j)));

        prev_A->at(j) = current_A->at(j);
        prev_b->at(j) = current_b->at(j);
        current_A->at(j) = layers->at(j).GetMatrixParams();
        current_b->at(j) = layers->at(j).GetVectorParams();
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

    std::vector<Matrix> prev_A(layers_count);
    std::vector<Vector> prev_b(layers_count);
    std::vector<Matrix> current_A(layers_count);
    std::vector<Vector> current_b(layers_count);
    InitializeParamsBuffers(layers, &prev_A, &prev_b, &current_A, &current_b);

    std::cout << "Training: Stochastic Gradient Descent With Momentum" << std::endl;

    size_t iter_count = 0;
    while (iter_count < max_iter_count) {
        std::shuffle(perm.begin(), perm.end(), mt);

        std::vector<Matrix> grads_A(layers_count);
        std::vector<Vector> grads_b(layers_count);

        CalculateGradientsOnBatch(&layers, &grads_A, &grads_b, error_block, perm, train_input,
                                  train_output);

        UpdateLayerParams(&layers, grads_A, grads_b, &prev_A, &prev_b, &current_A, &current_b, n);

        ++iter_count;
    }
}
}  // namespace NeuralNetworkApp
