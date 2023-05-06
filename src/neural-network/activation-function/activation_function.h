#pragma once

#include <functional>
#include <cmath>
#include <cassert>

namespace NeuralNetworkApp {

enum FunctionType { Sigmoid, Relu };

struct FunctionList {
    static std::function<double(double)> function(FunctionType type) {
        switch (type) {
            case Sigmoid:
                return [](double x) -> double { return 1 / (1 + exp(-x)); };
            case Relu:
                return [](double x) -> double { return std::max(0.0, x); };
            default:
                assert(false);
                return [](double x) { return 0.; };
        }
    }

    static std::function<double(double)> derivative(FunctionType type) {
        switch (type) {
            case Sigmoid:
                return [](double x) -> double {
                    return (1 / (1 + exp(-x))) * (1 - (1 / (1 + exp(-x))));
                };
            case Relu:
                return [](double x) -> double { return x < 0 ? 0 : 1.0; };
            default:
                assert(false);
                return [](double x) { return 0.; };
        }
    }
};

class ActivationFunction {
    using Function = std::function<double(double)>;

public:
    ActivationFunction()
        : function_(FunctionList::function(FunctionType::Sigmoid)),
          derivative_(FunctionList::derivative(FunctionType::Sigmoid)) {
    }

    ActivationFunction(FunctionType type)
        : function_(FunctionList::function(type)), derivative_(FunctionList::derivative(type)) {
    }

    const Function& GetFunction() const;

    const Function& GetDerivative() const;

private:
    Function function_;
    Function derivative_;
};

}  // namespace NeuralNetworkApp
