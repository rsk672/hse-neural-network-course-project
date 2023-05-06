#include "application.h"

namespace NeuralNetworkApp {

Application::Application() {
    for (size_t i = 0; i < 200; ++i) {
        double x1 = rand() % 2;
        double x2 = rand() % 2;
        double y = (x1 != x2);

        train_input_.push_back({x1, x2});
        train_output_.push_back({y});
    }
}

void Application::Run1() {
    NeuralNetworkApp::NeuralNetwork network({2, 4, 4, 4, 1},
                                            {
                                                NeuralNetworkApp::FunctionType::Sigmoid,
                                                NeuralNetworkApp::FunctionType::Sigmoid,
                                                NeuralNetworkApp::FunctionType::Sigmoid,
                                                NeuralNetworkApp::FunctionType::Sigmoid,
                                            },
                                            NeuralNetworkApp::AdamOptimizer(),
                                            NeuralNetworkApp::ErrorType::MSE);

    network.Train(train_input_, train_output_, 10000);

    std::vector<double> prediction = network.Predict({0, 0});
    std::cout << "0 0 " << prediction[0] << "\n";
    prediction = network.Predict({0, 1});
    std::cout << "0 1 " << prediction[0] << "\n";
    prediction = network.Predict({1, 0});
    std::cout << "1 0 " << prediction[0] << "\n";
    prediction = network.Predict({1, 1});
    std::cout << "1 1 " << prediction[0] << "\n";
}

void Application::Run2() {
    NeuralNetworkApp::NeuralNetwork network({2, 4, 4, 4, 1},
                                            {
                                                NeuralNetworkApp::FunctionType::Sigmoid,
                                                NeuralNetworkApp::FunctionType::Sigmoid,
                                                NeuralNetworkApp::FunctionType::Sigmoid,
                                                NeuralNetworkApp::FunctionType::Sigmoid,
                                            },
                                            NeuralNetworkApp::SGDOptimizer(),
                                            NeuralNetworkApp::ErrorType::MSE);

    network.Train(train_input_, train_output_, 10000);
    std::vector<double> prediction = network.Predict({0, 0});
    std::cout << "0 0 " << prediction[0] << "\n";
    prediction = network.Predict({0, 1});
    std::cout << "0 1 " << prediction[0] << "\n";
    prediction = network.Predict({1, 0});
    std::cout << "1 0 " << prediction[0] << "\n";
    prediction = network.Predict({1, 1});
    std::cout << "1 1 " << prediction[0] << "\n";
}

}  // namespace NeuralNetworkApp