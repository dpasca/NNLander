#ifndef TRAININGTASKRES_H
#define TRAININGTASKRES_H

#include <random>
#include <vector>
#include <numeric>
#include <limits> // Needed for numeric_limits
#include <cmath>  // For std::sqrt, std::exp
#include <cassert> // For assert
#include <algorithm> // For std::transform, std::clamp
#include "Utils.h"
#include "SimpleNeuralNet.h"
#include "Simulation.h"

//==================================================================
// TrainingTaskRES class - handles neural network training using REINFORCE-ES
//==================================================================
class TrainingTaskRES
{
private:
    SimParams          mSimParams;
    std::vector<int>   mNetworkArchitecture;

    // Training parameters
    size_t             mMaxGenerations = 0;      // Maximum number of generations/updates
    size_t             mCurrentGeneration = 0;   // Current generation/update step
    double             mSigma = 0.1;             // Standard deviation for noise perturbation
    double             mAlpha = 0.01;            // Learning rate
    size_t             mNumPerturbations = 50;   // Number of perturbation pairs (N/2 in some literature)

    // Number of simulations to run for each perturbed network evaluation
    // More variants -> more accurate evaluation (helps prevent overfitting)
    static constexpr size_t SIM_VARIANTS_N = 10;

    // Central network being trained
    SimpleNeuralNet mCentralNetwork;
    double mBestScore = -std::numeric_limits<double>::max(); // Track the best score achieved by the central network

    // Random number generator
    std::mt19937 mRng;
    std::normal_distribution<float> mNoiseDistribution{0.0f, 1.0f}; // Standard normal distribution

    // Parameter vector size
    size_t mTotalParams = 0;

public:
    TrainingTaskRES(
        const SimParams& sp,
        const std::vector<int>& architecture,
        size_t maxGenerations,
        double sigma,
        double alpha,
        size_t numPerturbations,
        uint32_t seed = 1234)
        : mSimParams(sp)
        , mNetworkArchitecture(architecture)
        , mMaxGenerations(maxGenerations)
        , mSigma(sigma)
        , mAlpha(alpha)
        , mNumPerturbations(numPerturbations)
        , mCentralNetwork(architecture) // Initialize central network
        , mRng(seed)
    {
        // Initialize central network with random parameters
        mCentralNetwork.InitializeRandomParameters(mRng());
        mTotalParams = mCentralNetwork.GetTotalParameterCount();

        // Initial evaluation of the central network
        mBestScore = evaluateNetwork(mCentralNetwork);
    }

    //==================================================================
    // Flatten network parameters into a single vector
    //==================================================================
    std::vector<float> flattenParameters(const SimpleNeuralNet& net) const
    {
        const auto& layers = net.GetLayerParameters();
        std::vector<float> flatParams;
        flatParams.reserve(mTotalParams); // Reserve space

        for (const auto& layer : layers)
        {
            flatParams.insert(flatParams.end(), layer.weights.begin(), layer.weights.end());
            flatParams.insert(flatParams.end(), layer.biases.begin(), layer.biases.end());
        }
        assert(flatParams.size() == mTotalParams);
        return flatParams;
    }

    //==================================================================
    // Unflatten parameters from a vector into layer structure
    //==================================================================
    std::vector<LayerParameters> unflattenParameters(const std::vector<float>& flatParams) const
    {
        assert(flatParams.size() == mTotalParams);
        std::vector<LayerParameters> layers = mCentralNetwork.GetLayerParameters(); // Get structure
        size_t currentIdx = 0;

        for (auto& layer : layers)
        {
            size_t weightCount = layer.weights.size();
            std::copy(
                flatParams.begin() + currentIdx,
                flatParams.begin() + currentIdx + weightCount,
                layer.weights.begin());
            currentIdx += weightCount;

            size_t biasCount = layer.biases.size();
            std::copy(
                flatParams.begin() + currentIdx,
                flatParams.begin() + currentIdx + biasCount,
                layer.biases.begin());
            currentIdx += biasCount;
        }
        assert(currentIdx == mTotalParams);
        return layers;
    }

    //==================================================================
    // Evaluate fitness for a given network over multiple simulation variants
    //==================================================================
    double evaluateNetwork(const SimpleNeuralNet& net) const
    {
        const uint32_t simStartSeed = 1134; // Consistent starting seed for evaluation runs
        double totalScore = 0.0;

        for (size_t i = 0; i < SIM_VARIANTS_N; ++i)
        {
            const auto variantSeed = simStartSeed + (uint32_t)i;
            totalScore += TestNetworkOnSimulation(variantSeed, net);
        }
        return totalScore / (double)SIM_VARIANTS_N;
    }


    //==================================================================
    // Run a single training iteration (one ES update step)
    //==================================================================
    void RunIteration()
    {
        if (IsTrainingComplete()) return;

        std::vector<float> centralParams = flattenParameters(mCentralNetwork);
        std::vector<float> gradientEstimate(mTotalParams, 0.0f);

        // Store results from perturbations
        struct PerturbationResult
        {
            double fitness_plus {};
            double fitness_minus {};
            std::vector<float> epsilon;
        };
        std::vector<PerturbationResult> results(mNumPerturbations);

        ParallelTasks pt; // Parallelization system

        // --- Generate and Evaluate Perturbations ---
        for (size_t i = 0; i < mNumPerturbations; ++i)
        {
            // Generate noise vector epsilon
            std::vector<float> epsilon(mTotalParams);
            for(size_t j = 0; j < mTotalParams; ++j)
                epsilon[j] = mNoiseDistribution(mRng);

            // Store epsilon for later gradient calculation
            results[i].epsilon = epsilon; // Copy epsilon

            // Create perturbed parameters theta_plus and theta_minus
            std::vector<float> params_plus = centralParams;
            std::vector<float> params_minus = centralParams;
            for(size_t j = 0; j < mTotalParams; ++j)
            {
                float perturbation = (float)mSigma * epsilon[j];
                params_plus[j] += perturbation;
                params_minus[j] -= perturbation;
                // Optional: Clamp parameters if needed, though often omitted in ES
                // params_plus[j] = std::clamp(params_plus[j], -1.0f, 1.0f);
                // params_minus[j] = std::clamp(params_minus[j], -1.0f, 1.0f);
            }

            // --- Evaluate theta_plus ---
            pt.AddTask([this, params_plus = std::move(params_plus), i, &results]()
            {
                SimpleNeuralNet net_plus(mNetworkArchitecture);
                net_plus.SetLayerParameters(unflattenParameters(params_plus));
                results[i].fitness_plus = evaluateNetwork(net_plus);
            });

            // --- Evaluate theta_minus ---
             pt.AddTask([this, params_minus = std::move(params_minus), i, &results]()
             {
                SimpleNeuralNet net_minus(mNetworkArchitecture);
                net_minus.SetLayerParameters(unflattenParameters(params_minus));
                results[i].fitness_minus = evaluateNetwork(net_minus);
            });
        }

        // Wait for all evaluations to complete
        pt.WaitAll();

        // --- Calculate Gradient Estimate ---
        for (size_t i = 0; i < mNumPerturbations; ++i)
        {
            double fitness_diff = results[i].fitness_plus - results[i].fitness_minus;
            const auto& epsilon = results[i].epsilon;
            for (size_t j = 0; j < mTotalParams; ++j)
            {
                gradientEstimate[j] += (float)fitness_diff * epsilon[j];
            }
        }

        // --- Update Central Parameters ---
        double scaleFactor = mAlpha / (2.0 * mNumPerturbations * mSigma);
        for (size_t j = 0; j < mTotalParams; ++j)
        {
            centralParams[j] += (float)scaleFactor * gradientEstimate[j];
             // Optional: Clamp parameters
             // centralParams[j] = std::clamp(centralParams[j], -1.0f, 1.0f);
        }

        // Set updated parameters in the central network
        mCentralNetwork.SetLayerParameters(unflattenParameters(centralParams));

        // Evaluate the updated central network and update best score if improved
        double currentCentralScore = evaluateNetwork(mCentralNetwork);
        if (currentCentralScore > mBestScore) {
            mBestScore = currentCentralScore;
            // Could potentially save the best network parameters here if needed
        }

        // Increment the generation counter
        mCurrentGeneration += 1;
    }

    //==================================================================
    // Test a network on a simulation
    // - "seed" gives the simulation variant to test
    // - "net" is the network to test
    // Returns the score of the simulation with the given network
    // (Identical to the one in TrainingTaskGA)
    //==================================================================
    double TestNetworkOnSimulation(
        uint32_t simulationSeed,
        const SimpleNeuralNet& net) const
    {
        // Create a simulation with the given seed
        Simulation sim(mSimParams, simulationSeed);

        // Run the simulation until it ends, or 30 (virtual) seconds have passed
        while (!sim.IsSimulationComplete() && sim.GetElapsedTimeS() < 30.0)
        {
            // Step the simulation forward...
            sim.AnimateSim([&](const float* states, float* actions)
            {
                // states -> net -> actions
                net.FeedForward(states, actions);
            });
        }
        // Return the score of the simulation
        return sim.CalculateScore();
    }

    // Getters for training status
    size_t GetCurrentGeneration() const { return mCurrentGeneration; }
    size_t GetMaxGenerations() const { return mMaxGenerations; }
    double GetBestScore() const { return mBestScore; } // Returns the best score seen for the central network
    double GetSigma() const { return mSigma; }
    double GetAlpha() const { return mAlpha; }
    size_t GetNumPerturbations() const { return mNumPerturbations; }
    bool IsTrainingComplete() const { return mCurrentGeneration >= mMaxGenerations; }
    // Get the central network object
    const SimpleNeuralNet& GetCentralNetwork() const { return mCentralNetwork; }
};

#endif // TRAININGTASKRES_H
