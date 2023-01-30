#pragma once

#include <functional>
#include <cmath>

enum BasicActivationFunctions { Sigmoid, Relu };

class ActivationFunction {
public:
    std::function<double(double)> function;
    std::function<double(double)> derivative;

    ActivationFunction();

    ActivationFunction(BasicActivationFunctions type);

    ActivationFunction(std::function<double(double)> func, std::function<double(double)> der);
};