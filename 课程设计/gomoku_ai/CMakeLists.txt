cmake_minimum_required(VERSION 3.21)

project(gomoku_ai LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(GOMOKU_CORE_SOURCES
    src/board.c
    src/rules.c
    src/eval.c
    src/nn_model.c
    src/search.c
)

add_executable(gomoku_ai
    src/main.c
    src/console_ui.c
    ${GOMOKU_CORE_SOURCES}
)

target_include_directories(gomoku_ai PRIVATE include)

if(UNIX AND NOT APPLE)
    target_link_libraries(gomoku_ai PRIVATE m)
endif()

if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(gomoku_ai PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_library(gomoku_engine SHARED
    src/selfplay_api.c
    ${GOMOKU_CORE_SOURCES}
)

target_include_directories(gomoku_engine PRIVATE include)

if(UNIX AND NOT APPLE)
    target_link_libraries(gomoku_engine PRIVATE m)
endif()

if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(gomoku_engine PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(CTest)

if(BUILD_TESTING)
    add_executable(gomoku_tests
        tests/test_gomoku.c
        src/board.c
        src/rules.c
    )
    target_include_directories(gomoku_tests PRIVATE include)
    if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(gomoku_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME gomoku_tests COMMAND gomoku_tests)

    add_executable(gomoku_ai_tests
        tests/test_ai.c
        ${GOMOKU_CORE_SOURCES}
    )
    target_include_directories(gomoku_ai_tests PRIVATE include)
    if(UNIX AND NOT APPLE)
        target_link_libraries(gomoku_ai_tests PRIVATE m)
    endif()
    if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(gomoku_ai_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME gomoku_ai_tests COMMAND gomoku_ai_tests)
endif()
