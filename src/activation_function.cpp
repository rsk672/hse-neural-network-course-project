#include "activation_function.h"

namespace NeuralNetworkApp {

std::function<double(double)> ActivationFunction::GetFunction() const {
    return function_;
}

std::function<double(double)> ActivationFunction::GetDerivative() const {
    return derivative_;
}

}  // namespace NeuralNetworkApp
