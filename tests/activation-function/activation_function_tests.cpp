#include <neural-network/activation-function/activation_function.h>
#include <gtest/gtest.h>
#include "../utils.h"

namespace NeuralNetworkApp {

static constexpr double function_test_checks_count = 100;

double ExpectedSigmoid(double x) {
    return 1 / (1 + exp(-x));
}

TEST(SigmoidTests, SigmoidValueCorrectness) {
    ActivationFunction activation(FunctionType::Sigmoid);
    auto function = activation.GetFunction();
    for (size_t i = 0; i < function_test_checks_count; ++i) {
        double x = GetRandomDouble(-100, 100);
        double expected = ExpectedSigmoid(x);
        double actual = function(x);

        EXPECT_DOUBLE_EQ(expected, actual);
    }
}

TEST(SigmoidTests, SigmoidDerivativeCorrectness) {
    ActivationFunction activation(FunctionType::Sigmoid);
    auto derivative = activation.GetDerivative();
    for (size_t i = 0; i < function_test_checks_count; ++i) {
        double x = GetRandomDouble(-100, 100);
        double expected = ExpectedSigmoid(x) * (1 - ExpectedSigmoid(x));
        double actual = derivative(x);

        EXPECT_DOUBLE_EQ(expected, actual);
    }
}

TEST(ReluTests, ReluValueCorrectness) {
    ActivationFunction activation(FunctionType::Relu);
    auto function = activation.GetFunction();
    for (size_t i = 0; i < function_test_checks_count; ++i) {
        double x = GetRandomDouble(-100, 100);
        double expected = x > 0 ? x : 0;
        double actual = function(x);

        EXPECT_DOUBLE_EQ(expected, actual);
    }
}

TEST(ReluTests, ReluDerivativeCorrectness) {
    ActivationFunction activation(FunctionType::Relu);
    auto derivative = activation.GetDerivative();
    for (size_t i = 0; i < function_test_checks_count; ++i) {
        double x = GetRandomDouble(-100, 100);
        double expected = x > 0 ? 1 : 0;
        double actual = derivative(x);

        EXPECT_DOUBLE_EQ(expected, actual);
    }
}

TEST(LeakyReluTests, LeakyReluValueCorrectness) {
    ActivationFunction activation(FunctionType::LeakyRelu);
    auto function = activation.GetFunction();
    for (size_t i = 0; i < function_test_checks_count; ++i) {
        double x = GetRandomDouble(-100, 100);
        double expected = x > 0 ? x : 0.01 * x;
        double actual = function(x);

        EXPECT_DOUBLE_EQ(expected, actual);
    }
}

TEST(LeakyReluTests, LeakyReluDerivativeCorrectness) {
    ActivationFunction activation(FunctionType::LeakyRelu);
    auto derivative = activation.GetDerivative();
    for (size_t i = 0; i < function_test_checks_count; ++i) {
        double x = GetRandomDouble(-100, 100);
        double expected = x > 0 ? 1 : 0.01;
        double actual = derivative(x);

        EXPECT_DOUBLE_EQ(expected, actual);
    }
}

}  // namespace NeuralNetworkApp
