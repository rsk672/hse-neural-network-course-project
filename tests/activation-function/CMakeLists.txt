add_executable(activation_function_tests activation_function_tests.cpp ../../src/neural-network/activation-function/activation_function.cpp)

target_link_libraries(activation_function_tests
 PRIVATE
  GTest::GTest
  Eigen3::Eigen)

include_directories(../../src/)

gtest_discover_tests(activation_function_tests)
