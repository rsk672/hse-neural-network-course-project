#include <limits>
#include <iostream>
#include "gd_optimizer.h"

namespace NeuralNetworkApp {

void GDOptimizer::CalculateGradients(std::vector<Layer>* layers, std::vector<Matrix>* grads_A,
                                     std::vector<Vector>* grads_b, const ErrorBlock& error_block,
                                     const std::vector<Vector>& train_input,
                                     const std::vector<Vector>& train_output) const {
    size_t layers_count = layers->size();
    size_t n = train_input.size();

    for (size_t j = 0; j < layers_count; ++j) {
        grads_A->at(j) = Matrix(layers->at(j).GetOutputSize(), layers->at(j).GetInputSize());
        grads_b->at(j) = Vector(layers->at(j).GetOutputSize());
    }

    for (size_t i = 0; i < n; ++i) {

        auto x = train_input[i];
        for (size_t j = 0; j < layers_count; ++j) {
            x = layers->at(j).PushForward(x);
        }

        Vector u = error_block.GetGradientValue(x, train_output[i]);
        for (ssize_t j = layers_count - 1; j >= 0; --j) {
            u = layers->at(j).PushBackwards(u, &grads_A->at(j), &grads_b->at(j));
        }
    }

    for (size_t j = 0; j < layers_count; ++j) {
        layers->at(j).ShiftParams(-grads_A->at(j) * learning_speed_ / n,
                                  -grads_b->at(j) * learning_speed_ / n);
    }
}

void GDOptimizer::UpdateLayerParams(std::vector<Layer>* layers, const std::vector<Matrix>& grads_A,
                                    const std::vector<Vector>& grads_b,
                                    size_t vectors_count) const {
    size_t layers_count = layers->size();
    for (size_t j = 0; j < layers_count; ++j) {
        layers->at(j).ShiftParams(-grads_A[j] * learning_speed_ / vectors_count,
                                  -grads_b[j] * learning_speed_ / vectors_count);
    }
}

void GDOptimizer::Train(std::vector<Layer>& layers, const ErrorBlock& error_block,
                        const std::vector<Vector>& train_input,
                        const std::vector<Vector>& train_output, size_t max_iter_count) const {

    size_t layers_count = layers.size();
    size_t n = train_input.size();

    std::cout << "Training: Gradient Descent\n";

    size_t iter_count = 0;
    while (iter_count < max_iter_count) {
        std::vector<Matrix> grads_A(layers_count);
        std::vector<Vector> grads_b(layers_count);

        CalculateGradients(&layers, &grads_A, &grads_b, error_block, train_input, train_output);

        UpdateLayerParams(&layers, grads_A, grads_b, train_input.size());

        ++iter_count;
    }
}
}  // namespace NeuralNetworkApp
