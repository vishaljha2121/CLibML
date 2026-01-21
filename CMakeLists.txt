cmake_minimum_required(VERSION 3.31)
project(MLFramework C)

set(CMAKE_C_STANDARD 11)

add_executable(MLFramework
        src/main.c
        src/memory_mngmnt/arena.h
        src/memory_mngmnt/arena.c
        include/base.h
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
        src/model/train/train.h
        src/tensor/tensorNew.c
        include/tensorNew.h
        include/base_defs.h
        include/str.h
        src/mg/mg_impl.c
        src/mg/mg_arena.h
        include/os.h
        include/optimizers.h
        src/optimizers/optimizers.c
        src/os/os_mac.c
        src/tensor/tensor_internal.h
        include/err.h
        src/err.c

        src/str.c
        include/cost.h
        src/cost/cost.c
        include/layers.h
        src/layers/layers.c
        src/layers/layers_activation.c
        src/layers/layers_internal.h
        src/layers/layers_conv_2d.c
        src/layers/layers_dense.c
        src/layers/layers_dropout.c
        src/layers/layers_flatten.c
        src/layers/layers_input.c
        src/layers/layers_norm.c
        src/layers/layers_null.c
        src/layers/layers_pooling_2d.c
        src/layers/layers_reshape.c
        include/network.h
        src/network/network.c
        include/img.h
        src/tensor/tensor_backend_cpu.c
        src/examples/snake/snake_game.c
        src/examples/snake/snake_ai.c
        src/examples/snake/snake_main.c
        src/image/img.c)
add_compile_options(-O3 -march=native -Wall -Wextra)