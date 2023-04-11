#include "error_block.h"

namespace NeuralNetworkApp {

double ErrorBlock::GetErrorValue(const Vector& input, const Vector& expected) const {
    return error_func_(input, expected);
}

Vector ErrorBlock::GetGradientValue(const Vector& input, const Vector& expected) const {
    return gradient_(input, expected);
}

}  // namespace NeuralNetworkApp
