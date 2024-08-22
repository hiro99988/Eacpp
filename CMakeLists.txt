cmake_minimum_required(VERSION 3.16)
project(Eacpp CXX)

enable_testing()

set(SRC_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src")

# List of source files
set(SRC_FILES
    "${SRC_DIRECTORY}/Algorithms/MpMoead.cpp"

    "${SRC_DIRECTORY}/Crossovers/BinomialCrossover.cpp"

    "${SRC_DIRECTORY}/Decompositions/Tchebycheff.cpp"

    "${SRC_DIRECTORY}/Mutations/BitFlipMutation.cpp"
    "${SRC_DIRECTORY}/Mutations/PolynomialMutation.cpp"

    "${SRC_DIRECTORY}/Problems/LZ1.cpp"
    "${SRC_DIRECTORY}/Problems/LZBase.cpp"
    "${SRC_DIRECTORY}/Problems/ZDTBase.cpp"
    "${SRC_DIRECTORY}/Problems/ZDT1.cpp"

    "${SRC_DIRECTORY}/Rng/Rng.cpp"

    "${SRC_DIRECTORY}/Samplings/UniformRandomSampling.cpp"
)

add_library(Eacpp STATIC "${SRC_FILES}")

target_compile_features(Eacpp PUBLIC cxx_std_20)

find_package(MPI REQUIRED)
target_link_libraries(Eacpp MPI::MPI_CXX)

target_include_directories(Eacpp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/example")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/test")
