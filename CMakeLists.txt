project(neural-network-course-project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

find_package(Eigen3 REQUIRED)

set(SOURCES
    main.cpp
    src/application.cpp
    src/neural-network/activation-function/activation_function.cpp
    src/neural-network/error-block/error_block.cpp
    src/neural-network/layer/layer.cpp
    src/neural-network/neural_network.cpp
    src/neural-network/optimizers/gd/gd_optimizer.cpp
    src/neural-network/optimizers/sgd/sgd_optimizer.cpp
    src/neural-network/optimizers/adam/adam_optimizer.cpp
    src/neural-network/optimizers/sgd_momentum/sgd_momentum_optimizer.cpp
    src/mnist/mnist_reader.cpp
)

enable_testing()
add_subdirectory(src)
add_subdirectory(tests)

add_executable(project-demo ${SOURCES})
target_link_libraries(project-demo PRIVATE Eigen3::Eigen)
include_directories("${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/optimizers")
