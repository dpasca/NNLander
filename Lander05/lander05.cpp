#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <limits>
#include <chrono>

#include "raylib.h"
#include "rlgl.h"
#include "Simulation.h"
#include "SimulationDisplay.h"
#include "SimpleNeuralNet.h"
#include "TrainingTaskGA.h"
#include "DrawUI.h"
#include "TemplateFeedForward.hpp"

static const int SCREEN_WIDTH = 800;
static const int SCREEN_HEIGHT = 600;
static const float RESTART_DELAY = 2.0f;

// Number of training generations to run
static const int MAX_TRAINING_GENERATIONS = 10000;
// Size of population
static const int POPULATION_SIZE = 200;
// Mutation parameters
static const double MUTATION_RATE = 0.1;
static const double MUTATION_STRENGTH = 0.3;

//==================================================================
// Network configuration
//==================================================================
static constexpr std::array<int, 7> NETWORK_ARCHITECTURE = {
    SIM_BRAINSTATE_N,             // Input layer: simulation state variables
    (int)((double)SIM_BRAINSTATE_N*1.25), // Hidden layer
    (int)((double)SIM_BRAINSTATE_N*1.25*1.25), // Hidden layer
    (int)((double)SIM_BRAINSTATE_N*1.25*1.25*1.25), // Hidden layer
    (int)((double)SIM_BRAINSTATE_N*1.25*1.25), // Hidden layer
    (int)((double)SIM_BRAINSTATE_N*1.25), // Hidden layer
    SIM_BRAINACTION_N             // Output layer: actions (up, left, right)
};

// Forward declarations
static void drawUI(Simulation& sim, TrainingTaskGA<NETWORK_ARCHITECTURE>& trainingTask);

//==================================================================
// Main function
//==================================================================
int main()
{
    // Enable anti-aliasing (MSAA 4X)
    //SetConfigFlags(FLAG_MSAA_4X_HINT);

    // Initialize window
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Lunar Lander - Genetic Algorithm Training Demo");
    SetTargetFPS(60);

    // Setup the simulation parameters
    SimParams sp;
    sp.SCREEN_WIDTH = (float)SCREEN_WIDTH;
    sp.SCREEN_HEIGHT = (float)SCREEN_HEIGHT;

    // Create the simulation object with the parameters
    uint32_t seed = 1134; // Initial random seed
    Simulation sim(sp, seed);

    // Create the training task
    TrainingTaskGA<NETWORK_ARCHITECTURE> trainingTask(
        sp,
        MAX_TRAINING_GENERATIONS,
        POPULATION_SIZE,
        MUTATION_RATE,
        MUTATION_STRENGTH
    );

    float restartTimer = 0.0f;

    // Variables to track training time
    auto trainingStartTime = std::chrono::steady_clock::now();
    bool hasTrainingCompleted = false;

    // Main game loop
    while (!WindowShouldClose())
    {
        // Run training iterations in the background
        if (!trainingTask.IsTrainingComplete())
        {
            // Run a single generation per frame to avoid blocking the UI too much
            trainingTask.RunIteration();
        }
        else if (!hasTrainingCompleted)
        {
            // Training just completed
            auto trainingEndTime = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(trainingEndTime - trainingStartTime).count();
            printf("Training completed in %i seconds\n", (int)duration);
            hasTrainingCompleted = true;
        }

        // Auto-restart after landing or crashing
        if (sim.mLander.mStateIsLanded || sim.mLander.mStateIsCrashed)
        {
            restartTimer += GetFrameTime();
            if (restartTimer >= RESTART_DELAY || IsKeyPressed(KEY_SPACE))
            {
                // Reset the simulation, keep the same seed
                sim = Simulation(sp, seed);
                seed += 1;
                restartTimer = 0.0f;
            }
        }
        else
        {
            // Animate the simulation with the neural network brain
            const auto& bestParams = trainingTask.GetBestNetworkParameters();
            sim.AnimateSim([&](const Eigen::Vector<float, NETWORK_ARCHITECTURE.front()>& states, Eigen::Vector<float, NETWORK_ARCHITECTURE.back()>& actions)
            {
                // states -> testNet(bestParams) -> actions
                std::apply([&](const auto&... params) { ::FeedForward(states, actions, params...); }, bestParams);
            });
        }

        // Begin drawing
        BeginDrawing();

        ClearBackground(BLACK);
        // Allow any triangle to be drawn regardless of winding order
        rlDisableBackfaceCulling();

        // Draw the simulation
        DrawSim(sim);
        // Draw UI
        drawUI(sim, trainingTask);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}

//==================================================================
static void drawUI(Simulation& sim, TrainingTaskGA<NETWORK_ARCHITECTURE>& trainingTask)
{
    // Draw neural network visualization
    //if (!sim.mLander.mStateIsLanded && !sim.mLander.mStateIsCrashed)
    //{
    //    SimpleNeuralNet net(NETWORK_ARCHITECTURE);
    //    DrawNeuralNetwork(net, trainingTask.GetBestNetworkParameters());
    //}

    const int fsize = 20;

    DrawUIBase(sim, fsize, "ai");

    // Draw training information
    DrawUITrainingStatus(trainingTask.IsTrainingComplete(), fsize);

    DrawText(TextFormat("Generation: %i/%i",
                       (int)trainingTask.GetCurrentGeneration(),
                       (int)trainingTask.GetMaxGenerations()),
            SCREEN_WIDTH - 300, 40, fsize, WHITE);

    const double bestScore = trainingTask.GetBestScore();
    DrawText(TextFormat("Best Score: %.2f", bestScore),
            SCREEN_WIDTH - 300, 70, fsize, bestScore > 500.0f ? GREEN : ORANGE);

    DrawText(TextFormat("Population Size: %i",
                       (int)trainingTask.GetPopulationSize()),
            SCREEN_WIDTH - 300, 100, fsize, WHITE);
}
