add_executable(layer_tests layer_tests.cpp ../../src/neural-network/layer/layer.cpp ../../src/neural-network/activation-function/activation_function.cpp)

target_link_libraries(layer_tests
 PRIVATE
  GTest::GTest
  Eigen3::Eigen)

include_directories(../../src/)

gtest_discover_tests(layer_tests)
