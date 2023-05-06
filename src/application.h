#include "neural-network/neural_network.h"

namespace NeuralNetworkApp {

class Application {
public:
    Application();

    void Run1();

    void Run2();

private:
    std::vector<std::vector<double>> train_input_;
    std::vector<std::vector<double>> train_output_;
};
}  // namespace NeuralNetworkApp