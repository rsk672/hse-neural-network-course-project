#include <neural-network/layer/layer.h>
#include <neural-network/activation-function/activation_function.h>
#include <gtest/gtest.h>
#include <iostream>

namespace NeuralNetworkApp {

TEST(LayerTests, Basic) {
    Layer l(5, 4, FunctionType::Sigmoid);

    ActivationFunction activation(FunctionType::Sigmoid);
    auto function = activation.GetFunction();

    auto A = l.GetMatrixParams();
    auto b = l.GetVectorParams();

    Vector input = Vector::Random(5);
    Vector expected_forward = A * input + b;
    for (size_t i = 0; i < expected_forward.size(); ++i) {
        expected_forward[i] = function(expected_forward[i]);
    }

    Vector actual_forward = l.PushForward(input);

    EXPECT_EQ(expected_forward, actual_forward);

    Vector backward_input = Vector::Random(4);
    auto activation_matrix = ((A * input + b).unaryExpr(activation.GetDerivative())).asDiagonal();

    Vector expected_backward = backward_input.transpose() * activation_matrix * A;
    Vector actual_backward = l.PushBackwards(backward_input, NULL, NULL);

    EXPECT_EQ(expected_backward, actual_backward);
}

TEST(LayerTests, Multiple) {

    std::vector<size_t> sizes(10);
    std::vector<Layer> layers;
    for (size_t i = 0; i < sizes.size(); ++i) {
        sizes[i] = rand() % 20 + 1;
        if (i > 0) {
            layers.emplace_back(sizes[i - 1], sizes[i], FunctionType::Relu);
        }
    }

    ActivationFunction activation(FunctionType::Relu);
    auto func = activation.GetFunction();

    Vector input = Vector::Random(sizes[0]);
    std::vector<Vector> inputs_to_remember(layers.size());

    for (size_t i = 0; i < layers.size(); ++i) {
        inputs_to_remember[i] = input;

        Matrix A = layers[i].GetMatrixParams();
        Vector b = layers[i].GetVectorParams();

        Vector expected_input = A * input + b;
        for (size_t i = 0; i < expected_input.size(); ++i) {
            expected_input[i] = func(expected_input[i]);
        }

        Vector actual_input = layers[i].PushForward(input);

        ASSERT_EQ(expected_input, actual_input);

        input = actual_input;
    }

    Vector backward_input = Vector::Random(sizes.back());
    for (ssize_t i = layers.size() - 1; i >= 0; --i) {

        Matrix A = layers[i].GetMatrixParams();
        Vector b = layers[i].GetVectorParams();

        auto activation_matrix =
            ((A * inputs_to_remember[i] + b).unaryExpr(activation.GetDerivative())).asDiagonal();

        Vector expected_backward = backward_input.transpose() * activation_matrix * A;
        Vector actual_backward = layers[i].PushBackwards(backward_input, NULL, NULL);

        ASSERT_EQ(expected_backward, actual_backward);

        backward_input = actual_backward;
    }
}

}  // namespace NeuralNetworkApp
