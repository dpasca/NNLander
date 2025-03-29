#ifndef FIXITURE_HPP
#define FIXITURE_HPP

#ifdef TEST_FIXITURE
    #include <gtest/gtest.h>
#else
#ifdef BENCHMARK_FIXITURE
    #include <benchmark/benchmark.h>
#endif
#endif

#include <cstddef>
#include <cstdint>
#include <Eigen/Dense>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>
#include "TemplateFeedForward.hpp"
#include "SimpleNeuralNet.h"

template<NetArch auto netArch>
#ifdef TEST_FIXITURE
class FeedForwardTest : public testing::Test
#else
#ifdef BENCHMARK_FIXITURE
class FeedForwardBenchmarck : public benchmark::Fixture
#endif
#endif
{
public:
    #ifdef TEST_FIXITURE
    void SetUp() override
    #else
    #ifdef BENCHMARK_FIXITURE
    void SetUp(::benchmark::State& state) override
    #endif
    #endif
    {
        std::mt19937 mRng(1234);
        std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(2.0f));

        for (size_t i = 0; i < params1.size(); i++)
            params1[i] = dist(mRng);

        for (int i = 0; i < netArch.front(); i++)
            inputs1[i] = dist(mRng);

        uint32_t paramIdx = 0;
        fillNetParams<float, netArch>([&](int, int, int){ return params1[paramIdx++]; }, params2);

        for (int i = 0; i < netArch.front(); i++)
            inputs2(i) = inputs1[i];
    }

protected:
    SimpleNeuralNet net = std::vector<int>(netArch.begin(), netArch.end());
    std::vector<float> params1 = std::vector<float>(net.GetTotalParameters());
    float inputs1[netArch.front()] = {};
    float outputs1[netArch.back()] = {};

    NetParam<float, netArch> params2;
    Eigen::Vector<float, netArch.front()> inputs2;
    Eigen::Vector<float, netArch.back()> outputs2;
};

#endif // FIXITURE_HPP
