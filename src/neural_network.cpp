#include "neural_network.h"

NeuralNetwork::NeuralNetwork(std::initializer_list<size_t> layers_sizes, double learning_speed)
    : layers_sizes_(layers_sizes), learning_speed_(learning_speed) {
}

void NeuralNetwork::AddLayer(size_t layer_size) {
    layers_sizes_.emplace_back(layer_size);
}

void NeuralNetwork::SetActivationFunction(BasicActivationFunctions func) {
    activation_ = ActivationFunction(func);
}

void NeuralNetwork::SetActivationFunction(std::function<double(double)> func,
                                          std::function<double(double)> der) {
    activation_ = ActivationFunction(func, der);
}

void NeuralNetwork::Train(std::vector<std::vector<double>>& train_input,
                          std::vector<std::vector<double>>& train_output, size_t batch_index) {

    // creating layers
    for (size_t i = 1; i < layers_sizes_.size(); ++i) {
        layers_.emplace_back(layers_sizes_[i - 1], layers_sizes_[i], activation_);
    }

    // training
    for (size_t i = 0; i < train_input.size(); ++i) {
        Eigen::VectorXd x = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(train_input[i].data(),
                                                                          train_input[i].size());
        Eigen::VectorXd y = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(train_output[i].data(),
                                                                          train_output[i].size());

        // push forward to save inputs and calculate error function gradient
        for (auto it = layers_.begin(); it != layers_.end(); ++it) {
            x = it->PushForward(x);
        }

        // pushing backwards to calculate params gradients
        Eigen::VectorXd u = error_block_.GetDerivative(x, y);
        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
            u = it->PushBackwards(u);
        }

        // updating params
        if (i % batch_index == 0 || i == train_input.size() - 1) {
            for (auto it = layers_.begin(); it != layers_.end(); ++it) {
                it->UpdateParams(learning_speed_);
            }
        }
    }
}

std::vector<double> NeuralNetwork::Predict(std::vector<double> data) {
    Eigen::VectorXd x = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(data.data(), data.size());

    for (auto it = layers_.begin(); it != layers_.end(); ++it) {
        x = it->PushForward(x);
    }

    return std::vector<double>(x.data(), x.data() + x.size());
}