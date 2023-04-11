#include "optimizer.h"

namespace NeuralNetworkApp {

double Optimizer::GetAverageError(std::vector<Layer>& layers, const ErrorBlock& error_block,
                                  const std::vector<Vector>& train_input,
                                  const std::vector<Vector>& train_output) const {
    size_t layers_count = layers.size();
    size_t n = train_input.size();
    double train_error = 0;
    for (size_t i = 0; i < n; ++i) {
        auto x = train_input[i];
        for (size_t j = 0; j < layers_count; ++j) {
            x = layers[j].PushForward(x);
        }
        train_error += error_block.GetErrorValue(x, train_output[i]);
    }

    return train_error / n;
}

}  // namespace NeuralNetworkApp

