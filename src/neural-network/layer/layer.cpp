#include "layer.h"
#include <iostream>

namespace NeuralNetworkApp {

Layer::Layer(size_t input_size, size_t output_size, FunctionType func)
    : A_(output_size, input_size), b_(output_size), activation_(func) {

    static std::mt19937 rng(std::random_device{}());
    static std::normal_distribution<> nd(0.0, sqrt(2.0 / (input_size + output_size)));

    A_ = A_.unaryExpr([](double dummy) { return nd(rng); });
}

Vector Layer::PushForward(const Vector& x) {
    input_ = x;
    Vector result = A_ * x + b_;
    result = result.unaryExpr(activation_.GetFunction());

    return result;
}

Vector Layer::PushForwardPredict(const Vector& x) const {
    Vector result = A_ * x + b_;
    result = result.unaryExpr(activation_.GetFunction());

    return result;
}

Vector Layer::PushBackwards(const Vector& u, Matrix* grad_A_curr, Vector* grad_b_curr) {

    Matrix sigma_deriatives_matrix =
        ((A_ * input_ + b_).unaryExpr(activation_.GetDerivative())).asDiagonal();

    // sigma_deriatives_matrix.transposeInPlace();

    Matrix grad_a = sigma_deriatives_matrix * u * input_.transpose();

    Vector grad_b = sigma_deriatives_matrix * u;

    Vector backward_u = u.transpose() * sigma_deriatives_matrix * A_;

    *grad_A_curr += grad_a;

    *grad_b_curr += grad_b;

    return backward_u;
}

void Layer::ShiftParams(const Matrix& A_shift, const Vector& b_shift) {
    A_ = A_ + A_shift;
    b_ = b_ + b_shift;
}

size_t Layer::GetInputSize() const {
    return A_.cols();
}

size_t Layer::GetOutputSize() const {
    return A_.rows();
}

}  // namespace NeuralNetworkApp
