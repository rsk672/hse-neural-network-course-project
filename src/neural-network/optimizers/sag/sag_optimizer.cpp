#include <limits>
#include <algorithm>
#include <random>
#include <vector>
#include <iostream>
#include "sag_optimizer.h"

namespace NeuralNetworkApp {

void SAGOptimizer::InitializeStoredGradientBuffers(const std::vector<Layer>& layers,
                                     std::vector<std::vector<Matrix>>* grads_A_all,
                                     std::vector<std::vector<Vector>>* grads_b_all) const {
    size_t vectors_count = grads_A_all->size();
    size_t layers_count = layers.size();
    for (size_t i = 0; i < vectors_count; ++i) {
        for (size_t j = 0; j < layers_count; ++j) {
            grads_A_all->at(i)[j] = Matrix(layers[j].GetOutputSize(), layers[j].GetInputSize());
            grads_b_all->at(i)[j] = Vector(layers[j].GetOutputSize());
        }
    }
}

void SAGOptimizer::InitializeGradientSumBuffers(const std::vector<Layer>& layers,
                                  std::vector<Matrix>* sum_grads_A,
                                  std::vector<Vector>* sum_grads_b) const {
    size_t layers_count = layers.size();
    for (size_t i = 0; i < layers_count; ++i) {
        sum_grads_A->at(i) = Matrix(layers[i].GetOutputSize(), layers[i].GetInputSize());
        sum_grads_b->at(i) = Vector(layers[i].GetOutputSize());
    }
}

void SAGOptimizer::CalculateGradientOnVector(std::vector<Layer>* layers, std::vector<Matrix>* grads_A,
                               std::vector<Vector>* grads_b, const ErrorBlock& error_block,
                               const std::vector<Vector>& train_input,
                               const std::vector<Vector>& train_output, size_t vector_index) const {

    size_t layers_count = layers->size();
    for (size_t i = 0; i < layers_count; ++i) {
        grads_A->at(i) = Matrix(layers->at(i).GetOutputSize(), layers->at(i).GetInputSize());
        grads_b->at(i) = Vector(layers->at(i).GetOutputSize());
    }

    auto x = train_input[vector_index];
    for (size_t i = 0; i < layers_count; ++i) {
        x = layers->at(i).PushForward(x);
    }

    Vector u = error_block.GetGradientValue(x, train_output[vector_index]);
    for (ssize_t i = layers_count - 1; i >= 0; --i) {
        u = layers->at(i).PushBackwards(u, &grads_A->at(i), &grads_b->at(i));
    }
}

void SAGOptimizer::UpdateGradientSum(std::vector<Matrix>* sum_grads_A, std::vector<Vector>* sum_grads_b,
                       const std::vector<std::vector<Matrix>>& grads_A_all,
                       const std::vector<std::vector<Vector>>& grads_b_all,
                       const std::vector<Matrix>& grads_A, const std::vector<Vector>& grads_b,
                       size_t random_index) const {
    size_t count = sum_grads_A->size();
    for (size_t i = 0; i < count; ++i) {
        sum_grads_A->at(i) = sum_grads_A->at(i) - grads_A_all[random_index][i] + grads_A[i];
        sum_grads_b->at(i) = sum_grads_b->at(i) - grads_b_all[random_index][i] + grads_b[i];
    }
}

void SAGOptimizer::UpdateLayerParams(std::vector<Layer>* layers, const std::vector<Matrix>& sum_grads_A,
                       const std::vector<Vector>& sum_grads_b, size_t encountered_count) const {
    size_t layers_count = layers->size();
    for (size_t i = 0; i < layers_count; ++i) {
        layers->at(i).ShiftParams(-sum_grads_A[i] * learning_speed_ / encountered_count,
                                  -sum_grads_b[i] * learning_speed_ / encountered_count);
    }
}

void SAGOptimizer::Train(std::vector<Layer>& layers, const ErrorBlock& error_block,
                         const std::vector<Vector>& train_input,
                         const std::vector<Vector>& train_output, size_t max_iter_count) const {

    static std::random_device rd;
    static std::mt19937 mt(rd());

    size_t layers_count = layers.size();
    size_t n = train_input.size();

    std::vector<std::vector<Matrix>> grads_A_all(n, std::vector<Matrix>(layers_count));
    std::vector<std::vector<Vector>> grads_b_all(n, std::vector<Vector>(layers_count));
    std::vector<Matrix> sum_grads_A(layers_count);
    std::vector<Vector> sum_grads_b(layers_count);

    InitializeStoredGradientBuffers(layers, &grads_A_all, &grads_b_all);
    InitializeGradientSumBuffers(layers, &sum_grads_A, &sum_grads_b);

    std::vector<bool> encountered(n);
    size_t encountered_count = 0;

    std::cout << "Training: SAG" << std::endl;

    size_t iter_count = 0;
    while (iter_count < max_iter_count) {

        size_t random_index = mt() % n;
        if (!encountered[random_index]) {
            encountered[random_index] = true;
            ++encountered_count;
        }

        std::vector<Matrix> grads_A(layers_count);
        std::vector<Vector> grads_b(layers_count);

        CalculateGradientOnVector(&layers, &grads_A, &grads_b, error_block, train_input,
                                  train_output, random_index);

        UpdateGradientSum(&sum_grads_A, &sum_grads_b, grads_A_all, grads_b_all, grads_A, grads_b,
                          random_index);

        grads_A_all[random_index] = grads_A;
        grads_b_all[random_index] = grads_b;

        UpdateLayerParams(&layers, sum_grads_A, sum_grads_b, encountered_count);

        ++iter_count;
    }
}
}  // namespace NeuralNetworkApp
