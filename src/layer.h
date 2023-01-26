#pragma once

#include <random>
#include <Eigen/Dense>
#include "activation_function.h"

class Layer {

public:
    Layer(size_t input_size, size_t output_size, ActivationFunction activation);

    Eigen::VectorXd PushForward(Eigen::VectorXd& x);

    Eigen::VectorXd PushBackwards(Eigen::VectorXd& u);

    void UpdateParams(double speed);

private:
    Eigen::MatrixXd A_;  // layer params
    Eigen::VectorXd b_;  // layer params
    ActivationFunction activation_;
    Eigen::VectorXd input_;
    Eigen::MatrixXd sum_grad_A_;
    Eigen::VectorXd sum_grad_b_;

    void InitializeParams(size_t n, size_t m);
};