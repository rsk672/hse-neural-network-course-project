#pragma once

#include <initializer_list>
#include <vector>
#include "layer.h"
#include "error_block.h"

class NeuralNetwork {

public:
    NeuralNetwork(std::initializer_list<size_t> layers_sizes, double learning_speed = 1.0);

    void AddLayer(size_t layer_size);

    void SetActivationFunction(BasicActivationFunctions func);

    void SetActivationFunction(std::function<double(double)> func,
                               std::function<double(double)> der);

    void Train(std::vector<std::vector<double>>& train_input,
               std::vector<std::vector<double>>& train_output, size_t batch_index = 1);

    std::vector<double> Predict(std::vector<double> data);

private:
    std::vector<size_t> layers_sizes_;
    std::vector<Layer> layers_;
    ActivationFunction activation_;
    ErrorBlock error_block_;
    double learning_speed_;
};