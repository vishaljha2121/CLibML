cmake_minimum_required(VERSION 3.20)
project(MNIST_Classifier LANGUAGES C)

set(CMAKE_C_STANDARD 11)

# Find MLFramework
if(NOT TARGET mlframework)
    find_package(MLFramework REQUIRED)
    set(MLFRAMEWORK_TARGET MLFramework::mlframework)
else()
    set(MLFRAMEWORK_TARGET mlframework)
endif()

# MNIST executable
add_executable(mnist_classifier
    src/mnist_main.c
)

target_link_libraries(mnist_classifier ${MLFRAMEWORK_TARGET})

# Install
install(TARGETS mnist_classifier
    RUNTIME DESTINATION bin
)
