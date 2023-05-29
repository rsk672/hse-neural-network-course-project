#include "neural-network/neural_network.h"
#include "mnist/mnist_reader.h"

namespace NeuralNetworkApp {

class Application {
public:
    Application();

    void Run1();

    void Run2();

    void Run3();

    void Run4();

    void Run5();

private:
    std::vector<std::vector<double>> xor_train_input_;
    std::vector<std::vector<double>> xor_train_output_;

    std::vector<double> mnist_train_labels_;
    std::vector<double> mnist_test_labels_;
    std::vector<std::vector<double>> mnist_train_images_;
    std::vector<std::vector<double>> mnist_test_images_;

    void InitializeXORData();
    void InitializeMNISTData();
    std::vector<std::vector<double>> NormalizeMNISTImages(
        const std::vector<std::vector<double>>& images);
    std::vector<std::vector<double>> OneHotEncodeMNISTLabels(
        const std::vector<double>& train_labels);
    void CalculateMNISTAccuracy(const std::vector<std::vector<double>>& images,
                                const std::vector<double>& labels,
                                const NeuralNetwork& network) const;
};
}  // namespace NeuralNetworkApp
