#include "trainingBenchmark.hpp"
#include "benchmark/benchmark.h"
#include "../Lander04/TrainingTaskGA.h"
#include <vector>

static void trainingTask04(benchmark::State& state)
{
    SimParams sp;
    TrainingTaskGA trainingTask(
        sp,
        std::vector(NETWORK_ARCHITECTURE.begin(), NETWORK_ARCHITECTURE.end()),
        MAX_TRAINING_GENERATIONS,
        POPULATION_SIZE,
        MUTATION_RATE,
        MUTATION_STRENGTH
    );
    SimpleNeuralNet testNet(std::vector(NETWORK_ARCHITECTURE.begin(), NETWORK_ARCHITECTURE.end()));

    for (auto _ : state)
    {
        trainingTask.RunIteration();
    }

    state.SetItemsProcessed(state.iterations());
    state.counters["iterations_per_second"] = benchmark::Counter(state.iterations(), benchmark::Counter::kIsRate);
}
BENCHMARK(trainingTask04);


