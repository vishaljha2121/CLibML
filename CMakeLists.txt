cmake_minimum_required(VERSION 3.31)
project(MLFramework C)

set(CMAKE_C_STANDARD 11)

add_executable(MLFramework
        src/main.c
        src/memory_mngmnt/arena.h
        src/memory_mngmnt/arena.c
        src/base.h
        src/tensor/tensor.c
        src/tensor/tensor.h
        src/random_generators/prng.c
        src/random_generators/prng.h
        src/autograd/autograd.c
        src/autograd/autograd.h
        src/model/variables/modelVariables.c
        src/model/variables/modelVariables.h
        src/model/program/modelProgram.c
        src/model/program/modelProgram.h
        src/model/train/train.c
        src/model/train/train.h)
add_compile_options(-O3 -march=native -Wall -Wextra)