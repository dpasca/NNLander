#ifndef TRAININGTASKGA_H
#define TRAININGTASKGA_H

#include <cstddef>
#include <random>
#include <algorithm>
#include <numeric>
#include <future>
#include <thread>
#include <tuple>
#include "TemplateFeedForward.hpp"
#include "Simulation.h"

#define USE_XAVIER_INIT 1
#define USE_MUTATION_STDDEV 0

//==================================================================
// ParallelTasks class - handles parallel execution of tasks
//==================================================================
class ParallelTasks
{
    std::vector<std::future<void>> mFutures;
    unsigned int mThreadsN {};
public:
    ParallelTasks() : mThreadsN(std::thread::hardware_concurrency()) {}

    void AddTask(std::function<void()> task)
    {
        if (mFutures.size() >= mThreadsN)
        {
            mFutures.front().wait();
            mFutures.erase(mFutures.begin());
        }
        mFutures.push_back(std::async(std::launch::async, task));
    }
};

//==================================================================
// Individual class - represents a single member of the population
//==================================================================
template<NetArch auto netArch>
class Individual
{
public:
    NetParam<float, netArch> parameters;  // Neural network parameters
    double fitness = -std::numeric_limits<double>::max();  // Fitness score (-infinity by default)

    Individual() = default;

    Individual(const NetParam<float, netArch>& params, double fitnessScore = -std::numeric_limits<double>::max())
        : parameters(params), fitness(fitnessScore) {}

    // Comparison operator for sorting by fitness (descending order)
    bool operator<(const Individual& other) const {
        return fitness > other.fitness;
    }
};

//==================================================================
// TrainingTaskGA class - handles neural network training using genetic algorithms
//==================================================================
template<NetArch auto netArch>
class TrainingTaskGA
{
private:
    using Individual = Individual<netArch>;
private:
    SimParams          mSimParams;

    // Training parameters
    size_t             mMaxGenerations = 0;              // Maximum number of generations
    size_t             mPopulationSize = 50;             // Size of population
    size_t             mCurrentGeneration = 0;           // Current generation
    double             mMutationRate = 0.1;              // Probability of mutation
    double             mMutationStrength = 0.3;          // Scale of mutation
    double             mElitePercentage = 0.1;           // Percentage of top individuals to keep unchanged
    // Number of simulations to run for each individual
    // More variants -> more accurate evaluation (helps prevent overfitting)
    static constexpr size_t SIM_VARIANTS_N = 20;

    // Population
    std::vector<Individual> mPopulation;
    Individual mBestIndividual;

    // Random number generator
    std::mt19937 mRng;

public:
    TrainingTaskGA(
        const SimParams& sp,
        size_t maxGenerations,
        size_t populationSize,
        double mutationRate,
        double mutationStrength,
        uint32_t seed = 1234)
        : mSimParams(sp)
        , mMaxGenerations(maxGenerations)
        , mPopulationSize(populationSize)
        , mMutationRate(mutationRate)
        , mMutationStrength(mutationStrength)
        , mRng(seed)
    {
        // Here we create the initial population, with random parameters

        // Create a work buffer for random parameters
        NetParam<float, netArch> paramsWorkBuff;

        // Generate random parameters for each individual
#if USE_XAVIER_INIT
        std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(2.0f));
#else
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
#endif
        for (size_t i=0; i < mPopulationSize; ++i)
        {
            // Generate random parameters in the work buffer
            fillNetParams<float, netArch>([&](int,int,int){ return dist(mRng); }, paramsWorkBuff);

            // Create an individual with the generated parameters
            mPopulation.emplace_back(paramsWorkBuff);
        }
    }

    //==================================================================
    // Run a single training iteration (one generation)
    void RunIteration()
    {
        // Create the next generation (if this is not the first generation)
        if (mCurrentGeneration != 0)
            evolve();

        // Evaluate the fitness of the population
        evaluatePopulation();

        // Sort the population by fitness (descending)
        std::sort(mPopulation.begin(), mPopulation.end());

        // Update best individual if necessary
        if (mPopulation[0].fitness > mBestIndividual.fitness)
        {
            mBestIndividual = mPopulation[0];
        }

        // Increment the generation counter
        mCurrentGeneration += 1;
    }

    //==================================================================
    // Evaluate fitness for all individuals in the population
    void evaluatePopulation()
    {
        const uint32_t simStartSeed = 1134;

        // General network object (invariant for all individuals)

        ParallelTasks pt; // Parallelization system

        // Evaluate each individual's fitness in parallel
        for (auto& individual : mPopulation)
        {
            pt.AddTask([&]()
            {
                double scores[SIM_VARIANTS_N] = { 0.0 };
                for (size_t i = 0; i < SIM_VARIANTS_N; ++i)
                {
                    const auto variantSeed = simStartSeed + (uint32_t)i;
                    scores[i] = TestNetworkOnSimulation(variantSeed, individual.parameters);
                }
                std::sort(scores, scores+SIM_VARIANTS_N);
                individual.fitness = scores[SIM_VARIANTS_N/2];
            });
        }
    }

    //==================================================================
    // Create a new generation through selection, crossover and mutation
    void evolve()
    {
        // Keep track of the original population
        std::vector<Individual> oldPopulation = mPopulation;

        // Clear the population for the new generation
        mPopulation.clear();

        // Calculate number of elite individuals to keep unchanged
        const size_t eliteCount = static_cast<size_t>(mPopulationSize * mElitePercentage);

        // Keep elite individuals
        for (size_t i = 0; i < eliteCount && i < oldPopulation.size(); ++i)
        {
            mPopulation.push_back(oldPopulation[i]);
        }

        // Fill the rest of the population with offspring from crossover
        while (mPopulation.size() < mPopulationSize)
        {
            // Select two parents
            const auto& parent1 = SelectParent(oldPopulation);
            const auto& parent2 = SelectParent(oldPopulation);

            // Perform crossover
            Individual child = Crossover(parent1, parent2);

            // Perform mutation
            mutate(child);

            // Add the child to the new population
            mPopulation.push_back(child);
        }
    }

    //==================================================================
    // Select a parent using tournament selection
    const Individual& SelectParent(const std::vector<Individual>& population)
    {
        // Number of individuals to consider in the tournament
        const size_t tournamentSize = 3;

        // Select random individuals for the tournament
        std::vector<size_t> indices(population.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), mRng);

        // Find the best individual among those selected
        size_t bestIdx = indices[0];
        for (size_t i = 1; i < tournamentSize && i < indices.size(); ++i)
        {
            if (population[indices[i]].fitness > population[bestIdx].fitness)
                bestIdx = indices[i];
        }

        return population[bestIdx];
    }

    //==================================================================
    // Crossover two parents to create a child
    Individual Crossover(const Individual& parent1, const Individual& parent2)
    {
        // Uniform crossover: each parameter has a 50% chance of coming from each parent
        NetParam<float, netArch> childParams;
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        
        auto crossoverLayer = [&]<size_t Idx>() {
            auto& childLayer = std::get<Idx>(childParams);
            auto& parent1Layer = std::get<Idx>(parent1.parameters);
            auto& parent2Layer = std::get<Idx>(parent2.parameters);
            for (int r = 0; r < std::remove_cvref_t<decltype(childLayer)>::RowsAtCompileTime; r++) {
                for (int c = 0; c < std::remove_cvref_t<decltype(childLayer)>::ColsAtCompileTime; c++) {
                    if (dist(mRng) < 0.5f)
                        childLayer(r, c) = parent1Layer(r, c);
                    else
                        childLayer(r, c) = parent2Layer(r, c);
                }
            }    
        };

        [&]<size_t... Idxs>(std::index_sequence<Idxs...>) {
            (crossoverLayer.template operator()<Idxs>(), ...);
        }(std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<decltype(childParams)>>>{});

        return Individual(childParams);
    }

    //==================================================================
    static std::array<float, 2> calcMeanAndStdDev(const std::vector<float>& params)
    {
        auto mean = std::accumulate(params.begin(), params.end(), 0.0f) / params.size();
        auto stdDev = std::sqrt(
                        std::accumulate(params.begin(), params.end(), 0.0f,
                            [mean](float sum, float x) {
                                return sum + (x - mean) * (x - mean);
                            }) / params.size());
        return { mean, stdDev };
    }

    //==================================================================
    // Mutate an individual
    void mutate(Individual& individual)
    {
        std::uniform_real_distribution<float> shouldMutate(0.0f, 1.0f);
#if USE_MUTATION_STDDEV
        auto [mean, stdDev] = calcMeanAndStdDev(individual.parameters);
        std::normal_distribution<float> mutation(mean, stdDev);
#else
        std::normal_distribution<float> mutation(0.0f, (float)mMutationStrength);
#endif
        auto mutateLayer = [&]<size_t Idx>() {
            auto& layer = std::get<Idx>(individual.parameters);
            for (int r = 0; r < std::remove_cvref_t<decltype(layer)>::RowsAtCompileTime; r++) {
                for (int c = 0; c < std::remove_cvref_t<decltype(layer)>::ColsAtCompileTime; c++) {
                    // Each parameter has a chance to mutate
                    if (shouldMutate(mRng) < mMutationRate)
                    {
                        // Add a normally distributed random value
                        layer(r, c) += mutation(mRng);
                        // Clamp to [-1, 1] range
                        layer(r, c) = std::clamp(layer(r, c), -1.0f, 1.0f);
                    }
                }
            }
        };
        [&]<size_t... Idxs>(std::index_sequence<Idxs...>) {
            (mutateLayer.template operator()<Idxs>(), ...);
        }(std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<decltype(individual.parameters)>>>{});
    }

    //==================================================================
    // Test a network on a simulation
    // - "seed" gives the simulation variant to test
    // - "params" are the weights and biases of the network to test
    // Returns the score of the simulation with the given parameters
    //==================================================================
    double TestNetworkOnSimulation(
        uint32_t simulationSeed,
        const NetParam<float, netArch>& params) const
    {
        // Create a simulation with the given seed
        Simulation sim(mSimParams, simulationSeed);

        // Run the simulation until it ends, or 30 (virtual) seconds have passed
        while (!sim.IsSimulationComplete() && sim.GetElapsedTimeS() < 30.0)
        {
            // Step the simulation forward...
            sim.AnimateSim([&](const Eigen::Vector<float, netArch.front()>& states, Eigen::Vector<float, netArch.back()>& actions)
            {
                // states -> net(params) -> actions
                std::apply([&](const auto&... params) { ::FeedForward(states, actions, params...); }, params);
            });
        }
        // Return the score of the simulation
        return sim.CalculateScore();
    }

    // Getters for training status
    size_t GetCurrentGeneration() const { return mCurrentGeneration; }
    size_t GetMaxGenerations() const { return mMaxGenerations; }
    double GetBestScore() const { return mBestIndividual.fitness; }
    size_t GetPopulationSize() const { return mPopulationSize; }
    bool IsTrainingComplete() const { return mCurrentGeneration >= mMaxGenerations; }
    const NetParam<float, netArch>& GetBestNetworkParameters() const { return mBestIndividual.parameters; }
    const std::vector<Individual>& GetPopulation() const { return mPopulation; }
};

#endif
