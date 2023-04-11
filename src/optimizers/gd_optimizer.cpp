#include <limits>
#include <iostream>
#include "gd_optimizer.h"

namespace NeuralNetworkApp {
void GDOptimizer::Train(std::vector<Layer>& layers, const ErrorBlock& error_block,
                        const std::vector<Vector>& train_input,
                        const std::vector<Vector>& train_output, double max_error,
                        size_t max_iter_count) const {

    size_t layers_count = layers.size();
    double error = std::numeric_limits<double>::max();
    size_t iter_count = 0;
    size_t n = train_input.size();

    std::cout << "Training: Gradient Descent\n";

    while (error > max_error && iter_count < max_iter_count) {
        std::vector<Matrix> grads_A(layers_count);
        std::vector<Vector> grads_b(layers_count);

        for (size_t j = 0; j < layers_count; ++j) {
            grads_A[j] = Matrix(layers[j].GetOutputSize(), layers[j].GetInputSize());
            grads_b[j] = Vector(layers[j].GetOutputSize());
        }

        // we calculate gradients for params using every train_input entry (real gradient)
        for (size_t i = 0; i < n; ++i) {
            // pushing forward
            auto x = train_input[i];
            for (size_t j = 0; j < layers_count; ++j) {
                x = layers[j].PushForward(x);
            }

            // pushing backwards to calculate params gradients
            Vector u = error_block.GetGradientValue(x, train_output[i]);
            for (ssize_t j = layers_count - 1; j >= 0; --j) {
                u = layers[j].PushBackwards(u, grads_A[j], grads_b[j]);
            }
        }

        // and then update params by the real gradient
        for (size_t j = 0; j < layers_count; ++j) {
            layers[j].ShiftParams(-grads_A[j] * learning_speed_ / n,
                                  -grads_b[j] * learning_speed_ / n);
        }

        // calculating train error
        error = GetAverageError(layers, error_block, train_input, train_output);

        std::cout << "Iteration: " << iter_count << " Error: " << error << "\n";

        ++iter_count;
    }
}
}  // namespace NeuralNetworkApp
