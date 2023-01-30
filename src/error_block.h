#pragma once

#include <Eigen/Dense>
#include <functional>

class ErrorBlock {
public:
    Eigen::VectorXd GetDerivative(Eigen::VectorXd& input, Eigen::VectorXd& expected);

private:
    std::function<double(double)> derivative_;
};