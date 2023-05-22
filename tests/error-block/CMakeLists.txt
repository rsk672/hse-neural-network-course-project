add_executable(error_block_tests error_block_tests.cpp ../../src/neural-network/error-block/error_block.cpp)

target_link_libraries(error_block_tests
 PRIVATE
  GTest::GTest
  Eigen3::Eigen)

include_directories(../../src/)

gtest_discover_tests(error_block_tests)