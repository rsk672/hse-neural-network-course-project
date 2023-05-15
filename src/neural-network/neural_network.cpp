#include "neural_network.h"

namespace NeuralNetworkApp {

void NeuralNetwork::CreateLayers(const std::initializer_list<size_t>& layers_sizes,
                                 const std::initializer_list<FunctionType>& functions) {
    assert(layers_sizes.size() == functions.size() + 1);
    auto functions_it = functions.begin();
    for (auto it = layers_sizes.begin(); it + 1 != layers_sizes.end(); ++it) {
        layers_.emplace_back(*it, *(it + 1), *(functions_it++));
    }
}

std::vector<Vector> NeuralNetwork::TransformData(
    const std::vector<std::vector<double>>& data) const {
    std::vector<Vector> x;
    for (size_t i = 0; i < data.size(); ++i) {
        x.push_back(Eigen::Map<const Vector, Eigen::Unaligned>(data[i].data(), data[i].size()));
    }
    return x;
}

NeuralNetwork::NeuralNetwork(std::initializer_list<size_t> layers_sizes,
                             std::initializer_list<FunctionType> functions) {
    CreateLayers(layers_sizes, functions);
}

void NeuralNetwork::AddNextLayer(size_t input_size, size_t output_size, FunctionType func) {
    layers_.emplace_back(input_size, output_size, func);
}

void NeuralNetwork::SetError(ErrorType type) {
    error_block_ = ErrorBlock(type);
}

void NeuralNetwork::Train(const std::vector<std::vector<double>>& train_input,
                          const std::vector<std::vector<double>>& train_output,
                          size_t max_iter_count) {

    std::vector<Vector> x = TransformData(train_input);
    std::vector<Vector> y = TransformData(train_output);

    optimizer_->Train(layers_, error_block_, x, y, max_iter_count);
}

std::vector<double> NeuralNetwork::Predict(const std::vector<double>& data) const {
    Vector x(Eigen::Map<const Vector, Eigen::Unaligned>(data.data(), data.size()));

    for (auto it = layers_.begin(); it != layers_.end(); ++it) {
        x = it->PushForwardPredict(x);
    }

    return std::vector<double>(x.data(), x.data() + x.size());
}

}  // namespace NeuralNetworkApp
