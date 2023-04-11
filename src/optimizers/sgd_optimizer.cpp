#include "sgd_optimizer.h"
#include <limits>
#include <algorithm>
#include <random>
#include <vector>
#include <iostream>

namespace NeuralNetworkApp {
void SGDOptimizer::Train(std::vector<Layer>& layers, const ErrorBlock& error_block,
                         const std::vector<Vector>& train_input,
                         const std::vector<Vector>& train_output, double max_error,
                         size_t max_iter_count) const {

    size_t layers_count = layers.size();
    double error = std::numeric_limits<double>::max();
    size_t iter_count = 0;
    size_t n = train_input.size();

    std::vector<size_t> perm = std::vector<size_t>(n);
    for (size_t i = 0; i < n; ++i) {
        perm[i] = i;
    }

    std::random_device rd;
    std::mt19937 mt(rd());

    std::cout << "Training: Stochastic Gradient Descent\n";

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

        for (size_t i = 0; i < layers_count; ++i) {
            layers[i].ShiftParams(-grads_A[i] * learning_speed_ / n,
                                  -grads_b[i] * learning_speed_ / n);
        }

        // calculating train error
        error = GetAverageError(layers, error_block, train_input, train_output);

        std::cout << "Iteration: " << iter_count << " Error: " << error << "\n";

        ++iter_count;
    }
}
}  // namespace NeuralNetworkApp
