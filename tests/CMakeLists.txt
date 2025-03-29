include(FetchContent)
include(GoogleTest)

add_executable(NNLander_tests "matrix_multiplication_test.cpp" "FeedForward_test.cpp")
add_executable(NNLander_benchmark "FeedForward_benchmark.cpp" "trainingBenchmark04.cpp" "trainingBenchmark05.cpp")

FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
    GIT_SHALLOW    TRUE
    GIT_PROGRESS   TRUE
)
set(BUILD_GMOCK       OFF)
set(INSTALL_GTEST     OFF)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.9.2
    GIT_SHALLOW    TRUE
    GIT_PROGRESS   TRUE
)
set(BENCHMARK_ENABLE_TESTING     OFF)
set(BENCHMARK_ENABLE_INSTALL     OFF)
set(BENCHMARK_INSTALL_DOCS       OFF)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
set(BENCHMARK_USE_BUNDLED_GTEST  OFF)
FetchContent_MakeAvailable(benchmark)

target_link_libraries(NNLander_tests PRIVATE GTest::gtest_main)
target_link_libraries(NNLander_benchmark PRIVATE benchmark::benchmark_main raylib)

gtest_discover_tests(NNLander_tests)

