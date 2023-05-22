#pragma once

#include <random>
#include <Eigen/Dense>
#include "../activation-function/activation_function.h"
#include "global.h"

namespace NeuralNetworkApp {

class Layer {

public:
    Layer(size_t input_size, size_t output_size, FunctionType func);

    Vector PushForward(const Vector& x);

    Vector PushForwardPredict(const Vector& x) const;

    Vector PushBackwards(const Vector& u, Matrix* grad_A_curr, Vector* grad_b_curr);

    void ShiftParams(const Matrix& A_update, const Vector& b_update);

    size_t GetInputSize() const;

    size_t GetOutputSize() const;

    Matrix GetMatrixParams() const;
    Vector GetVectorParams() const;

private:
    Matrix A_;  // layer params
    Vector b_;  // layer params
    ActivationFunction activation_;
    Vector input_;
};

}  // namespace NeuralNetworkApp
