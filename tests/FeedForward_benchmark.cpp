#define BENCHMARK_FIXITURE

#include "fixiture.hpp"

BENCHMARK_TEMPLATE_F(FeedForwardBenchmarck, loop10x3, std::array<int, 2>{10, 3})(benchmark::State& st) {
    for (auto _ : st) {
        net.FeedForward(params1.data(), inputs1, outputs1);
    }
}

BENCHMARK_TEMPLATE_F(FeedForwardBenchmarck, matrix10x3, std::array<int, 2>{10, 3})(benchmark::State& st) {
    for (auto _ : st) {
        std::apply([&](const auto&... matrices) { ::FeedForward(inputs2, outputs2, matrices...); }, params2);
    }
}

BENCHMARK_TEMPLATE_F(FeedForwardBenchmarck, loop10x12x12x3, std::array<int, 4>{10, 12, 12, 3})(benchmark::State& st) {
    for (auto _ : st) {
        net.FeedForward(params1.data(), inputs1, outputs1);
    }
}

BENCHMARK_TEMPLATE_F(FeedForwardBenchmarck, matrix10x12x12x3, std::array<int, 4>{10, 12, 12, 3})(benchmark::State& st) {
    for (auto _ : st) {
        std::apply([&](const auto&... matrices) { ::FeedForward(inputs2, outputs2, matrices...); }, params2);
    }
}

BENCHMARK_TEMPLATE_F(FeedForwardBenchmarck, loop10x20x30x20x10, std::array<int, 5>{10, 20, 30, 20, 10})(benchmark::State& st) {
    for (auto _ : st) {
        net.FeedForward(params1.data(), inputs1, outputs1);
    }
}

BENCHMARK_TEMPLATE_F(FeedForwardBenchmarck, matrix10x20x30x20x10, std::array<int, 5>{10, 20, 30, 20, 10})(benchmark::State& st) {
    for (auto _ : st) {
        std::apply([&](const auto&... matrices) { ::FeedForward(inputs2, outputs2, matrices...); }, params2);
    }
}

BENCHMARK_TEMPLATE_F(FeedForwardBenchmarck, loop10x20x30x40x30x20x10, std::array<int, 7>{10, 20, 30, 40, 30, 20, 10})(benchmark::State& st) {
    for (auto _ : st) {
        net.FeedForward(params1.data(), inputs1, outputs1);
    }
}

BENCHMARK_TEMPLATE_F(FeedForwardBenchmarck, matrix10x20x30x40x30x20x10, std::array<int, 7>{10, 20, 30, 40, 30, 20, 10})(benchmark::State& st) {
    for (auto _ : st) {
        std::apply([&](const auto&... matrices) { ::FeedForward(inputs2, outputs2, matrices...); }, params2);
    }
}
