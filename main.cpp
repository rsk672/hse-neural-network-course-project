#include <iostream>
#include "src/neural_network.h"

int main() {

    std::vector<std::vector<double>> train_input{};
    std::vector<std::vector<double>> train_output{};

    for (size_t i = 0; i < 200; ++i) {
        int x1 = rand() % 2;
        int x2 = rand() % 2;
        int y = (x1 != x2);

        train_input.push_back({static_cast<double>(x1), static_cast<double>(x2)});
        train_output.push_back({static_cast<double>(y)});
    }

    NeuralNetworkApp::NeuralNetwork network({2, 4, 4, 4, 1},
                                            {
                                                NeuralNetworkApp::FunctionType::Sigmoid,
                                                NeuralNetworkApp::FunctionType::Sigmoid,
                                                NeuralNetworkApp::FunctionType::Sigmoid,
                                                NeuralNetworkApp::FunctionType::Sigmoid,
                                            });

    network.SetOptimizer(NeuralNetworkApp::OptimizerType::Adam);

    network.SetError(NeuralNetworkApp::ErrorType::MSE);

    network.Train(train_input, train_output, 0.001, 50000);

    std::vector<double> prediction = network.Predict({0, 0});
    std::cout << "0 0 " << prediction[0] << "\n";

    prediction = network.Predict({0, 1});
    std::cout << "0 1 " << prediction[0] << "\n";

    prediction = network.Predict({1, 0});
    std::cout << "1 0 " << prediction[0] << "\n";

    prediction = network.Predict({1, 1});
    std::cout << "1 1 " << prediction[0] << "\n";
}
