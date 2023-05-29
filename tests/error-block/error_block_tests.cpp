#include <neural-network/error-block/error_block.h>
#include <gtest/gtest.h>
#include "../utils.h"

namespace NeuralNetworkApp {

std::vector<double> GetRandomVector(size_t size) {
    std::vector<double> result(size);
    for (size_t i = 0; i < size; ++i) {
        result[i] = GetRandomDouble(-100, 100);
    }
    return result;
}

Vector TransformToEigen(const std::vector<double>& vec) {
    return Eigen::Map<const Vector, Eigen::Unaligned>(vec.data(), vec.size());
}

TEST(MSETests, MSEValueCorrectness) {
    ErrorBlock block(ErrorType::MSE);

    static constexpr double vector_size = 8;
    std::vector<double> v1 = GetRandomVector(vector_size);
    std::vector<double> v2 = GetRandomVector(vector_size);

    double expected = 0;
    for (size_t i = 0; i < vector_size; ++i) {
        expected += (v2[i] - v1[i]) * (v2[i] - v1[i]) / vector_size;
    }

    Vector v1_transformed = TransformToEigen(v1);
    Vector v2_transformed = TransformToEigen(v2);

    double actual = block.GetErrorValue(v1_transformed, v2_transformed);

    EXPECT_DOUBLE_EQ(expected, actual);
}

TEST(MSETests, MSEGradientCorrectness) {
    ErrorBlock block(ErrorType::MSE);

    static constexpr double vector_size = 8;
    std::vector<double> v1 = GetRandomVector(vector_size);
    std::vector<double> v2 = GetRandomVector(vector_size);

    std::vector<double> expected(vector_size);
    for (size_t i = 0; i < vector_size; ++i) {
        expected[i] = 2.0 * (v1[i] - v2[i]) / vector_size;
    }

    Vector v1_transformed = TransformToEigen(v1);
    Vector v2_transformed = TransformToEigen(v2);

    Vector actual = block.GetGradientValue(v1_transformed, v2_transformed);

    EXPECT_EQ(TransformToEigen(expected), actual);
}

TEST(MAETests, MAEValueCorrectness) {
    ErrorBlock block(ErrorType::MAE);

    static constexpr double vector_size = 8;
    std::vector<double> v1 = GetRandomVector(vector_size);
    std::vector<double> v2 = GetRandomVector(vector_size);

    double expected = 0;
    for (size_t i = 0; i < vector_size; ++i) {
        expected += fabs(v2[i] - v1[i]) / vector_size;
    }

    Vector v1_transformed = TransformToEigen(v1);
    Vector v2_transformed = TransformToEigen(v2);

    double actual = block.GetErrorValue(v1_transformed, v2_transformed);

    EXPECT_DOUBLE_EQ(expected, actual);
}

TEST(MAETests, MAEGradientCorrectness) {
    ErrorBlock block(ErrorType::MAE);

    static constexpr double vector_size = 8;
    std::vector<double> v1 = GetRandomVector(vector_size);
    std::vector<double> v2 = GetRandomVector(vector_size);

    std::vector<double> expected(vector_size);
    for (size_t i = 0; i < vector_size; ++i) {
        expected[i] = (v1[i] > v2[i] ? 1.0 : -1.0) / vector_size;
    }

    Vector v1_transformed = TransformToEigen(v1);
    Vector v2_transformed = TransformToEigen(v2);

    Vector actual = block.GetGradientValue(v1_transformed, v2_transformed);

    EXPECT_EQ(TransformToEigen(expected), actual);
}

}  // namespace NeuralNetworkApp
