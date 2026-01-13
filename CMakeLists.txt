cmake_minimum_required(VERSION 3.31)
project(MLFramework C)

set(CMAKE_C_STANDARD 11)

add_executable(MLFramework src/main.c
        src/memory_mngmnt/arena.h
        src/memory_mngmnt/arena.c
        src/base.h)
add_compile_options(-O3 -march=native -Wall -Wextra)