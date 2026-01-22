cmake_minimum_required(VERSION 3.20)
project(MLFramework VERSION 1.0.0 LANGUAGES C)

# Library build
add_subdirectory(lib)

# Examples (optional)
option(BUILD_EXAMPLES "Build example projects" ON)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Tests (optional)
option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()