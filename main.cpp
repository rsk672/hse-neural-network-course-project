#include <iostream>
#include "src/neural_network.h"

int main() {
    std::vector<std::vector<double>> train_input{};
    std::vector<std::vector<double>> train_output{};
    std::vector<std::vector<double>> test_data{};

    for (size_t i = 0; i < 2000; ++i) {
        bool x1 = rand() % 2;
        bool x2 = rand() % 2;
        bool y = x1 != x2;
        train_input.push_back({x1 * 1.0, x2 * 1.0});
        train_output.push_back({y * 1.0});
    }

    NeuralNetwork network({2, 4, 4, 1});
    network.SetActivationFunction(
        [](double x) { return 0.5 * tanh(x) + 0.5; },
        [](double x) { return 2.0 / ((exp(-x) + exp(x)) * (exp(-x) + exp(x))); });

    network.Train(train_input, train_output);

    for (size_t i = 0; i < 100; ++i) {
        bool x1 = rand() % 2;
        bool x2 = rand() % 2;

        std::vector<double> prediction = network.Predict({x1 * 1.0, x2 * 1.0});
        std::cout << x1 << " " << x2 << " " << prediction[0] << std::endl;
    }
}