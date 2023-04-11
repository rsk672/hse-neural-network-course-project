#include <limits>
#include <algorithm>
#include <random>
#include <vector>
#include <iostream>
#include "sag_optimizer.h"

namespace NeuralNetworkApp {
void SAGOptimizer::Train(std::vector<Layer>& layers, const ErrorBlock& error_block,
                         const std::vector<Vector>& train_input,
                         const std::vector<Vector>& train_output, double max_error,
                         size_t max_iter_count) const {

    std::random_device rd;
    std::mt19937 mt(rd());

    size_t layers_count = layers.size();
    double error = std::numeric_limits<double>::max();
    size_t iter_count = 0;
    size_t n = train_input.size();

    // store gradients for each train input
    std::vector<std::vector<Matrix>> grads_A_all(n, std::vector<Matrix>(layers_count));
    std::vector<std::vector<Vector>> grads_b_all(n, std::vector<Vector>(layers_count));

    // store gradients sum
    std::vector<Matrix> sum_grads_A(layers_count);
    std::vector<Vector> sum_grads_b(layers_count);

    // initialize all gradients and sums
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < layers_count; ++j) {
            grads_A_all[i][j] = Matrix(layers[j].GetOutputSize(), layers[j].GetInputSize());
            grads_b_all[i][j] = Vector(layers[j].GetOutputSize());

            sum_grads_A[j] = Matrix(layers[j].GetOutputSize(), layers[j].GetInputSize());
            sum_grads_b[j] = Vector(layers[j].GetOutputSize());
        }
    }

    std::vector<bool> encountered(n);
    size_t encountered_count = 0;

    std::cout << "Training: SAG\n";

    while (error > max_error && iter_count < max_iter_count) {
        size_t ix = mt() % n;
        if (!encountered[ix]) {
            encountered[ix] = true;
            ++encountered_count;
        }

        // calculating gradient on train_input[ix]

        std::vector<Matrix> grads_A(layers_count);
        std::vector<Vector> grads_b(layers_count);
        for (size_t j = 0; j < layers_count; ++j) {
            grads_A[j] = Matrix(layers[j].GetOutputSize(), layers[j].GetInputSize());
            grads_b[j] = Vector(layers[j].GetOutputSize());
        }

        auto x = train_input[ix];
        for (size_t j = 0; j < layers_count; ++j) {
            x = layers[j].PushForward(x);
        }

        Vector u = error_block.GetGradientValue(x, train_output[ix]);
        for (ssize_t j = layers_count - 1; j >= 0; --j) {
            u = layers[j].PushBackwards(u, grads_A[j], grads_b[j]);
        }

        // update current gradients using train_input[ix]

        for (size_t j = 0; j < layers_count; ++j) {
            sum_grads_A[j] = sum_grads_A[j] - grads_A_all[ix][j] + grads_A[j];
            sum_grads_b[j] = sum_grads_b[j] - grads_b_all[ix][j] + grads_b[j];
        }

        // update stored gradients

        grads_A_all[ix] = grads_A;
        grads_b_all[ix] = grads_b;

        // shift params using sum of gradients

        for (size_t j = 0; j < layers_count; ++j) {
            layers[j].ShiftParams(-sum_grads_A[j] * learning_speed_ / encountered_count,
                                  -sum_grads_b[j] * learning_speed_ / encountered_count);
        }

        // calculating train error

        error = GetAverageError(layers, error_block, train_input, train_output);

        std::cout << "Iteration: " << iter_count << " Error: " << error << "\n";

        ++iter_count;
    }
}
}  // namespace NeuralNetworkApp
