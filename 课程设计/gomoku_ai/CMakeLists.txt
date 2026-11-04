cmake_minimum_required(VERSION 3.21)

project(gomoku_ai LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

add_executable(gomoku_ai
    src/main.c
    src/function_list.c
)
