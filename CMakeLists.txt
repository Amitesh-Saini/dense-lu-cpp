cmake_minimum_required(VERSION 3.20)
project(lu_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Useful for VS Code IntelliSense / clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 REQUIRED CONFIG)

# ---- Core library ----
add_library(lu_lib
    src/lu.cpp
    src/verify.cpp
    # src/Matrix.cpp   # only if you create it later
)

target_include_directories(lu_lib
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

# ---- Example ----
add_executable(lu_example
    examples/solve_demo.cpp
)

target_link_libraries(lu_example
    PRIVATE lu_lib
)

# ---- Bench ----
add_executable(bench_lu
    bench/bench_lu.cpp
)

target_link_libraries(bench_lu
    PRIVATE lu_lib
)

# ---- Tests ----
option(BUILD_TESTS "Build unit tests" ON)

if(BUILD_TESTS)
    enable_testing()

    add_executable(test_lu
        tests/test_lu.cpp
        tests/test_lu_utils.cpp
    )

    target_link_libraries(test_lu
        PRIVATE lu_lib Eigen3::Eigen
    )

    add_test(NAME test_lu COMMAND test_lu)
endif()