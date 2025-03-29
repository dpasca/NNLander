#ifndef TRAININGBENCHMARK_HPP
#define TRAININGBENCHMARK_HPP

#include "Simulation.h"

constexpr int MAX_TRAINING_GENERATIONS = 10000;
constexpr int POPULATION_SIZE = 200;
constexpr double MUTATION_RATE = 0.1;
constexpr double MUTATION_STRENGTH = 0.3;
constexpr std::array<int, 10> NETWORK_ARCHITECTURE = {
    SIM_BRAINSTATE_N,
    (int)((double)SIM_BRAINSTATE_N * 1.25),
    (int)((double)SIM_BRAINSTATE_N * 1.25),
    (int)((double)SIM_BRAINSTATE_N * 1.25),
    (int)((double)SIM_BRAINSTATE_N * 1.25),
    (int)((double)SIM_BRAINSTATE_N * 1.25),
    (int)((double)SIM_BRAINSTATE_N * 1.25),
    (int)((double)SIM_BRAINSTATE_N * 1.25),
    (int)((double)SIM_BRAINSTATE_N * 1.25),
    SIM_BRAINACTION_N};

#endif // TRAININGBENCHMARK_HPP
