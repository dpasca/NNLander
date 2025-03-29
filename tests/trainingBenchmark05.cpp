#include "trainingBenchmark.hpp"
#include "benchmark/benchmark.h"
#include "../Lander05/TrainingTaskGA.h"

static void trainingTask05(benchmark::State& state)
{
    SimParams sp;
    TrainingTaskGA<NETWORK_ARCHITECTURE> trainingTask(
        sp,
        MAX_TRAINING_GENERATIONS,
        POPULATION_SIZE,
        MUTATION_RATE,
        MUTATION_STRENGTH
    );

    for (auto _ : state)
    {
        trainingTask.RunIteration();
    }

    state.SetItemsProcessed(state.iterations());
    state.counters["iterations_per_second"] = benchmark::Counter(state.iterations(), benchmark::Counter::kIsRate);
}
BENCHMARK(trainingTask05);
