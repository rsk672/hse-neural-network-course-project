#include "neural_network.h"

namespace NeuralNetworkApp {

NeuralNetwork::NeuralNetwork(std::initializer_list<size_t> layers_sizes,
                             std::initializer_list<FunctionType> functions) {
    auto functions_it = functions.begin();
    for (auto it = layers_sizes.begin(); it + 1 != layers_sizes.end(); ++it) {
        layers_.emplace_back(*it, *(it + 1), *(functions_it++));
    }
}

void NeuralNetwork::AddNextLayer(size_t input_size, size_t output_size, FunctionType func) {
    layers_.emplace_back(input_size, output_size, func);
}

void NeuralNetwork::SetError(ErrorType type) {
    error_block_ = ErrorBlock(type);
}

void NeuralNetwork::Train(const std::vector<std::vector<double>>& train_input,
                          const std::vector<std::vector<double>>& train_output, double max_error,
                          size_t max_iter_count) {

    std::vector<Vector> x;
    std::vector<Vector> y;

    for (size_t i = 0; i < train_input.size(); ++i) {
        x.push_back(Eigen::Map<const Vector, Eigen::Unaligned>(train_input[i].data(),
                                                               train_input[i].size()));
    }

    for (size_t i = 0; i < train_output.size(); ++i) {
        y.push_back(Eigen::Map<const Vector, Eigen::Unaligned>(train_output[i].data(),
                                                               train_output[i].size()));
    }

    if (optimizer_) {
        optimizer_->Train(layers_, error_block_, x, y, max_error, max_iter_count);
    }
}

std::vector<double> NeuralNetwork::Predict(const std::vector<double>& data) {
    Vector x(Eigen::Map<const Vector, Eigen::Unaligned>(data.data(), data.size()));

    for (auto it = layers_.begin(); it != layers_.end(); ++it) {
        x = it->PushForward(x);
    }

    return std::vector<double>(x.data(), x.data() + x.size());
}

}  // namespace NeuralNetworkApp
