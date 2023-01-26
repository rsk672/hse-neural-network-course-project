#include "activation_function.h"

ActivationFunction::ActivationFunction() {
    function = [](double x) { return 1 / (1 + exp(-x)); };
    derivative = [=](double x) { return (1 / (1 + exp(-x))) * (1 - (1 / (1 + exp(-x)))); };
}

ActivationFunction::ActivationFunction(BasicActivationFunctions type) {
    if (type == BasicActivationFunctions::Sigmoid) {
        function = [](double x) { return 1 / (1 + exp(-x)); };
        derivative = [](double x) { return (1 / (1 + exp(-x))) * (1 - (1 / (1 + exp(-x)))); };
    }
    if (type == BasicActivationFunctions::Relu) {
        function = [](double x) { return std::max(0.0, x); };
        derivative = [](double x) { return x < 0 ? 0.0 : 1.0; };
    }
}

ActivationFunction::ActivationFunction(std::function<double(double)> func,
                                       std::function<double(double)> der) {
    function = func;
    derivative = der;
}