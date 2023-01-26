#include "error_block.h"

Eigen::VectorXd ErrorBlock::GetDerivative(Eigen::VectorXd& input, Eigen::VectorXd& expected) {
    Eigen::VectorXd result = -2 * (expected - input);
    return result;
}