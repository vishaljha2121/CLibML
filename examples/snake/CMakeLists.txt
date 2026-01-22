cmake_minimum_required(VERSION 3.20)
project(SnakeDQN LANGUAGES C)

set(CMAKE_C_STANDARD 11)

# Find MLFramework (either installed or from parent build)
if(NOT TARGET mlframework)
    find_package(MLFramework REQUIRED)
    set(MLFRAMEWORK_TARGET MLFramework::mlframework)
else()
    set(MLFRAMEWORK_TARGET mlframework)
endif()

# Snake executable
add_executable(snake_dqn
    src/snake_main.c
    src/snake_game.c
    src/snake_ai.c
)

target_link_libraries(snake_dqn ${MLFRAMEWORK_TARGET})

# Install
install(TARGETS snake_dqn
    RUNTIME DESTINATION bin
)
