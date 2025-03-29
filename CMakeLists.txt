cmake_minimum_required(VERSION 3.10)
include(FetchContent)

option(NNL_BUILD_TESTS "Build test executable" OFF)

project(LanderSimulation)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FETCHCONTENT_QUIET FALSE)

# Set optimization flags for all build types
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ob2 /Oi /Ot /GL")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -flto")
endif()

# Specify output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Find raylib package if installed system-wide
find_package(raylib 5.5 EXACT QUIET)

if (raylib_FOUND)
    message("using raylib found at ${raylib_LIBRARY}")
else (NOT raylib_FOUND)
    # If system raylib not found, fetch and build it
    FetchContent_Declare(
        raylib
        GIT_REPOSITORY https://github.com/raysan5/raylib.git
        GIT_TAG 5.5
        GIT_SHALLOW ON
        GIT_PROGRESS TRUE
    )
    # Set raylib build options
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build raylib as a static library")
    set(RAYLIB_BUILD_MODE Release CACHE STRING "Build mode for raylib")
    FetchContent_MakeAvailable(raylib)
endif()

FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)
# Make the headers available without building
FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
  FetchContent_Populate(eigen)
endif()

# Add the common headers directory
include_directories(Common)

include_directories(${eigen_SOURCE_DIR})

# Add subdirectories for each lander demo
add_subdirectory(Lander01)
add_subdirectory(Lander02)
add_subdirectory(Lander03)
add_subdirectory(Lander04)
add_subdirectory(Lander05)

if (NNL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
