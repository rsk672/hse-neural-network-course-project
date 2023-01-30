#include "layer.h"

Layer::Layer(size_t input_size, size_t output_size, ActivationFunction activation) {
    activation_ = activation;
    InitializeParams(input_size, output_size);
}

Eigen::VectorXd Layer::PushForward(Eigen::VectorXd& x) {
    input_ = x;  // save input when pushing forward
    Eigen::VectorXd result = A_ * x + b_;
    result = result.unaryExpr(activation_.function);

    return result;
}

Eigen::VectorXd Layer::PushBackwards(Eigen::VectorXd& u) {
    Eigen::MatrixXd sigma_deriatives_matrix =
        (A_ * input_ + b_).unaryExpr(activation_.derivative).asDiagonal();

    Eigen::MatrixXd grad_a = sigma_deriatives_matrix.transpose() * u * input_.transpose();
    Eigen::VectorXd grad_b = sigma_deriatives_matrix.transpose() * u;

    Eigen::VectorXd backward_u = u.transpose() * sigma_deriatives_matrix * A_;

    sum_grad_A_ += grad_a;
    sum_grad_b_ += grad_b;

    return backward_u;
}

void Layer::UpdateParams(double speed) {
    A_ -= sum_grad_A_ * speed;
    b_ -= sum_grad_b_ * speed;
    sum_grad_A_ = Eigen::MatrixXd::Zero(A_.rows(), A_.cols());
    sum_grad_b_ = Eigen::VectorXd::Zero(b_.size());
}

void Layer::InitializeParams(size_t n, size_t m) {
    A_ = Eigen::MatrixXd(m, n);
    b_ = Eigen::VectorXd(m);

    sum_grad_A_ = Eigen::MatrixXd(m, n);
    sum_grad_b_ = Eigen::VectorXd(m);

    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> nd{0, 1};
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            A_(i, j) = nd(gen);
        }
        b_[i] = nd(gen);
    }
}