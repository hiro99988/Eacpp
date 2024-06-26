cmake_minimum_required(VERSION 3.16)
project(Eacpp CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/out")

set(SRC_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src")

# List of source files
set(SRC_FILES
    "${SRC_DIRECTORY}/Crossovers/CrossoverBase.cpp"
    "${SRC_DIRECTORY}/Crossovers/OnePointCrossover.cpp"

    "${SRC_DIRECTORY}/Decompositions/Tchebycheff.cpp"

    "${SRC_DIRECTORY}/Mutations/BitFlipMutation.cpp"

    "${SRC_DIRECTORY}/Problems/LZ1.cpp"
    "${SRC_DIRECTORY}/Problems/LZBase.cpp"
    "${SRC_DIRECTORY}/Samplings/UniformRandomSampling.cpp"

    "${SRC_DIRECTORY}/Selections/RandomSelection.cpp"

    "${SRC_DIRECTORY}/Utils/Rng.cpp"
)

add_library(Eacpp STATIC "${SRC_FILES}")

target_compile_features(Eacpp PUBLIC cxx_std_20)

target_include_directories(Eacpp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/example")
