#include <limits>
#include <algorithm>
#include <random>
#include <vector>
#include <iostream>
#include "adam_optimizer.h"

namespace NeuralNetworkApp {
void AdamOptimizer::Train(std::vector<Layer>& layers, const ErrorBlock& error_block,
                          const std::vector<Vector>& train_input,
                          const std::vector<Vector>& train_output, double max_error,
                          size_t max_iter_count) const {

    std::random_device rd;
    std::mt19937 mt(rd());

    size_t layers_count = layers.size();
    double error = std::numeric_limits<double>::max();
    size_t iter_count = 0;
    size_t n = train_input.size();
    std::vector<size_t> perm = std::vector<size_t>(n);
    for (size_t i = 0; i < n; ++i) {
        perm[i] = i;
    }

    // initialize buffers for current first and second moments (m - first moment, v - second moment)
    std::vector<Matrix> current_m_A(layers_count);
    std::vector<Vector> current_m_b(layers_count);
    std::vector<Matrix> current_v_A(layers_count);
    std::vector<Vector> current_v_b(layers_count);
    for (size_t j = 0; j < layers_count; ++j) {
        current_m_A[j] = Matrix(layers[j].GetOutputSize(), layers[j].GetInputSize());
        current_m_b[j] = Vector(layers[j].GetOutputSize());
        current_v_A[j] = Matrix(layers[j].GetOutputSize(), layers[j].GetInputSize());
        current_v_b[j] = Vector(layers[j].GetOutputSize());
    }

    double current_learning_speed = learning_speed_;

    std::cout << "Training: Adam\n";

    while (error > max_error && iter_count < max_iter_count) {

        std::shuffle(perm.begin(), perm.end(),
                     mt);  // shuffling vector indices to choose random vectors for batch

        std::vector<Matrix> grads_A(layers_count);
        std::vector<Vector> grads_b(layers_count);
        for (size_t j = 0; j < layers_count; ++j) {
            grads_A[j] = Matrix(layers[j].GetOutputSize(), layers[j].GetInputSize());
            grads_b[j] = Vector(layers[j].GetOutputSize());
        }

        // we calculate gradients using random vectors from train input
        for (size_t i = 0; i < std::min(n, batch_size_); ++i) {

            // pushing forward
            auto x = train_input[perm[i]];
            for (size_t j = 0; j < layers_count; ++j) {
                x = layers[j].PushForward(x);
            }

            // pushing backwards to calculate params gradients
            Vector u = error_block.GetGradientValue(x, train_output[perm[i]]);

            for (ssize_t j = layers_count - 1; j >= 0; --j) {
                u = layers[j].PushBackwards(u, grads_A[j], grads_b[j]);
            }
        }

        // update first and second moments
        for (size_t i = 0; i < layers_count; ++i) {
            current_m_A[i] = beta1_ * current_m_A[i] + (1 - beta1_) * grads_A[i];
            current_m_b[i] = beta1_ * current_m_b[i] + (1 - beta1_) * grads_b[i];

            current_v_A[i] = beta2_ * current_v_A[i] +
                             (1 - beta2_) * grads_A[i].unaryExpr([](double x) { return x * x; });
            current_v_b[i] = beta2_ * current_v_b[i] +
                             (1 - beta2_) * grads_b[i].unaryExpr([](double x) { return x * x; });
        }

        // updating layers params using moments
        current_learning_speed = learning_speed_ * sqrt(1 - pow(beta2_, iter_count + 1)) /
                                 (1 - pow(beta1_, iter_count + 1));
        for (size_t i = 0; i < layers_count; ++i) {
            Matrix A_shift = (-current_learning_speed * current_m_A[i].array() /
                              (current_v_A[i].array().sqrt() - eps_))
                                 .matrix();
            Vector b_shift = (-current_learning_speed * current_m_b[i].array() /
                              (current_v_b[i].array().sqrt() - eps_));

            layers[i].ShiftParams(A_shift, b_shift);
        }

        // calculating train error
        error = GetAverageError(layers, error_block, train_input, train_output);

        std::cout << "Iteration: " << iter_count << " Error: " << error << "\n";

        ++iter_count;
    }
}
}  // namespace NeuralNetworkApp
