#pragma once

#include <Eigen/Dense>
#include <functional>
#include <cassert>
#include "global.h"

namespace NeuralNetworkApp {

enum ErrorType { MSE, MAE };

struct ErrorList {
    static std::function<double(const Vector&, const Vector&)> Error(ErrorType type) {
        switch (type) {
            case MSE:
                return [](const Vector& input, const Vector& expected) -> double {
                    return (input - expected).dot(input - expected) / input.size();
                };
            case MAE:
                return [](const Vector& input, const Vector& expected) -> double {
                    return (input - expected)
                        .unaryExpr([](double x) { return std::fabs(x); })
                        .mean();
                };
            default:
                assert(false);
                return [](const Vector& input, const Vector& expected) { return 0.; };
        }
    }

    static std::function<Vector(const Vector&, const Vector&)> Gradient(ErrorType type) {
        switch (type) {
            case MSE:
                return [](const Vector& input, const Vector& expected) -> Vector {
                    return 2 * (input - expected) / input.size();
                };
            case MAE:
                return [](const Vector& input, const Vector& expected) -> Vector {
                    return (input - expected).unaryExpr([](double x) -> double {
                        return x > 0 ? 1.0 : -1.0;
                    }) / (input.size());
                };
            default:
                assert(false);
                return
                    [](const Vector& input, const Vector& expected) { return (input - expected); };
        }
    }
};

class ErrorBlock {
public:
    ErrorBlock(ErrorType type)
        : error_func_(ErrorList::Error(type)), gradient_(ErrorList::Gradient(type)) {
    }

    double GetErrorValue(const Vector& input, const Vector& expected) const;

    Vector GetGradientValue(const Vector& input, const Vector& expected) const;

private:
    std::function<double(const Vector&, const Vector&)> error_func_;
    std::function<Vector(const Vector&, const Vector&)> gradient_;
};

}  // namespace NeuralNetworkApp
