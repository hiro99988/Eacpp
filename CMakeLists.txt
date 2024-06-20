cmake_minimum_required(VERSION 3.20)
project(Eacpp CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/out")

set(SRC_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src")

# List of source files
set(SRC_FILES
    "${SRC_DIRECTORY}/Crossovers/CrossoverBase.cpp"
    "${SRC_DIRECTORY}/Crossovers/OnePointCrossover.cpp"

    "${SRC_DIRECTORY}/Utils/Rng.cpp"
)

add_library(Eacpp STATIC "${SRC_FILES}")

target_compile_features(Eacpp PUBLIC cxx_std_20)

target_include_directories(Eacpp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/example")
