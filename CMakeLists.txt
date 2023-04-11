project(neural-network-course-project)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

find_package(Eigen3 REQUIRED)

set(SOURCES
    main.cpp
    src/activation_function.cpp
    src/error_block.cpp
    src/layer.cpp
    src/neural_network.cpp
    src/optimizers/optimizer.cpp
    src/optimizers/gd_optimizer.cpp
    src/optimizers/sgd_optimizer.cpp
    src/optimizers/sag_optimizer.cpp
    src/optimizers/adam_optimizer.cpp
)

add_executable(project-demo ${SOURCES})
target_link_libraries(project-demo PRIVATE Eigen3::Eigen)
include_directories("${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/optimizers")
