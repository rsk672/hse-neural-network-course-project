project(neural-network-course-project)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

find_package(Eigen3 REQUIRED)

set(SOURCES
    main.cpp
    src/activation_function.cpp
    src/error_block.cpp
    src/layer.cpp
    src/neural_network.cpp
)

add_executable(project-demo ${SOURCES})
target_link_libraries(project-demo PRIVATE Eigen3::Eigen)

