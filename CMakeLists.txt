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
)

target_include_directories(lu_lib
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

# ---- Examples ----
add_executable(solve_demo
    examples/solve_demo.cpp
)

target_link_libraries(solve_demo
    PRIVATE
        lu_lib
)

add_executable(steady_heat_1d
    examples/steady_heat_1d.cpp
)

target_link_libraries(steady_heat_1d
    PRIVATE
        lu_lib
)

# ---- Bench ----
add_executable(bench_lu
    bench/bench_lu.cpp

    bench/bench_common.cpp
    bench/bench_generators.cpp
    bench/bench_custom_lu.cpp
    bench/bench_multiple_rhs.cpp
    bench/bench_residuals.cpp
    bench/bench_eigen_utils.cpp
    bench/bench_csv.cpp

    tests/test_lu_utils.cpp
)

target_include_directories(bench_lu
    PRIVATE
        ${CMAKE_SOURCE_DIR}/bench
        ${CMAKE_SOURCE_DIR}/tests
)

target_link_libraries(bench_lu
    PRIVATE
        lu_lib
        Eigen3::Eigen
)

# Windows memory measurement uses GetProcessMemoryInfo from Psapi.
if(WIN32)
    target_link_libraries(bench_lu
        PRIVATE
            Psapi
    )
endif()

# ---- Theoretical memory CSV generator ----
add_executable(bench_make_theoretical_memory
    bench/bench_make_theoretical_memory.cpp
)

target_include_directories(bench_make_theoretical_memory
    PRIVATE
        ${CMAKE_SOURCE_DIR}/bench
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
        PRIVATE
            lu_lib
            Eigen3::Eigen
    )

    add_test(NAME test_lu COMMAND test_lu)
endif()