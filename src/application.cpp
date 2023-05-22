#include "application.h"
#include <filesystem>

namespace NeuralNetworkApp {

Application::Application() {
    InitializeXORData();
    InitializeMNISTData();
}

void Application::InitializeXORData() {
    for (size_t i = 0; i < 200; ++i) {
        double x1 = rand() % 2;
        double x2 = rand() % 2;
        double y = (x1 != x2);

        xor_train_input_.push_back({x1, x2});
        xor_train_output_.push_back({y});
    }
}

std::vector<std::vector<double>> Application::NormalizeMNISTImages(
    const std::vector<std::vector<double>>& images) {
    std::vector<std::vector<double>> normalized_images =
        std::vector<std::vector<double>>(images.size());
    for (size_t i = 0; i < normalized_images.size(); ++i) {
        for (size_t j = 0; j < images[i].size(); ++j) {
            normalized_images[i].push_back(images[i][j] / 255.0);
        }
    }

    return normalized_images;
}

std::vector<std::vector<double>> Application::OneHotEncodeMNISTLabels(
    const std::vector<double>& labels) {
    std::vector<std::vector<double>> encoded_labels =
        std::vector<std::vector<double>>(labels.size(), std::vector<double>(10));
    for (size_t i = 0; i < encoded_labels.size(); ++i) {
        encoded_labels[i][labels[i]] = 1;
    }

    return encoded_labels;
}

void Application::InitializeMNISTData() {
    MNISTReader train_reader("../src/mnist/data/train-images.idx3-ubyte",
                             "../src/mnist/data/train-labels.idx1-ubyte");
    MNISTReader test_reader("../src/mnist/data/t10k-images.idx3-ubyte",
                            "../src/mnist/data/t10k-labels.idx1-ubyte");

    mnist_train_labels_ = train_reader.GetLabelsData();
    mnist_train_images_ = train_reader.GetImagesData();
    mnist_test_labels_ = test_reader.GetLabelsData();
    mnist_test_images_ = test_reader.GetImagesData();
}

void Application::Run1() {
    NeuralNetwork network({2, 4, 4, 4, 1},
                          {
                              FunctionType::Sigmoid,
                              FunctionType::Sigmoid,
                              FunctionType::Sigmoid,
                              FunctionType::Sigmoid,
                          },
                          AdamOptimizer(0.001, 0.9, 0.999, 10e-8, 2), ErrorType::MSE);

    network.Train(xor_train_input_, xor_train_output_, 10000);

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
    NeuralNetwork network({2, 4, 4, 4, 1},
                          {
                              FunctionType::Sigmoid,
                              FunctionType::Sigmoid,
                              FunctionType::Sigmoid,
                              FunctionType::Sigmoid,
                          },
                          SGDOptimizer(16, 5.0), ErrorType::MSE);

    network.Train(xor_train_input_, xor_train_output_, 10000);

    std::vector<double> prediction = network.Predict({0, 0});
    std::cout << "0 0 " << prediction[0] << "\n";
    prediction = network.Predict({0, 1});
    std::cout << "0 1 " << prediction[0] << "\n";
    prediction = network.Predict({1, 0});
    std::cout << "1 0 " << prediction[0] << "\n";
    prediction = network.Predict({1, 1});
    std::cout << "1 1 " << prediction[0] << "\n";
}

void Application::Run3() {
    NeuralNetwork network({2, 4, 4, 4, 1},
                          {
                              FunctionType::LeakyRelu,
                              FunctionType::Sigmoid,
                              FunctionType::Sigmoid,
                              FunctionType::LeakyRelu,
                          },
                          SGDMomentumOptimizer(16, 2.0, 0.90), ErrorType::MSE);

    network.Train(xor_train_input_, xor_train_output_, 10000);

    std::vector<double> prediction = network.Predict({0, 0});
    std::cout << "0 0 " << prediction[0] << "\n";
    prediction = network.Predict({0, 1});
    std::cout << "0 1 " << prediction[0] << "\n";
    prediction = network.Predict({1, 0});
    std::cout << "1 0 " << prediction[0] << "\n";
    prediction = network.Predict({1, 1});
    std::cout << "1 1 " << prediction[0] << "\n";
}

void Application::CalculateMNISTAccuracy(const std::vector<std::vector<double>>& images,
                                         const std::vector<double>& labels,
                                         const NeuralNetwork& network) const {
    double predicted_cnt = 0;
    for (size_t i = 0; i < images.size(); ++i) {
        std::vector<double> predicted = network.Predict(images[i]);
        std::cout << "Actual: " << labels[i] << " Predicted: ";
        double max = 0;
        size_t max_ix = 0;
        for (size_t j = 0; j < predicted.size(); ++j) {
            if (predicted[j] > max) {
                max = predicted[j];
                max_ix = j;
            }
        }
        std::cout << max_ix << "\n";
        if (max_ix == labels[i]) {
            ++predicted_cnt;
        }
    }

    std::cout << "Accuracy: " << predicted_cnt / labels.size();
}

void Application::Run4() {

    auto train_input = NormalizeMNISTImages(mnist_train_images_);
    auto train_output = OneHotEncodeMNISTLabels(mnist_train_labels_);

    NeuralNetwork network({784, 256, 10},
                          {
                              FunctionType::Relu,
                              FunctionType::Sigmoid,
                          },
                          AdamOptimizer(), ErrorType::MSE);

    network.Train(train_input, train_output, 4001);

    auto test_images_normalized = NormalizeMNISTImages(mnist_test_images_);
    CalculateMNISTAccuracy(test_images_normalized, mnist_test_labels_, network);
}

}  // namespace NeuralNetworkApp