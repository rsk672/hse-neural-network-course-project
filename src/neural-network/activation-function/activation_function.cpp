#include "activation_function.h"

namespace NeuralNetworkApp {

const ActivationFunction::Function& ActivationFunction::GetFunction() const {
    return function_;
}

const ActivationFunction::Function& ActivationFunction::GetDerivative() const {
    return derivative_;
}

}  // namespace NeuralNetworkApp
