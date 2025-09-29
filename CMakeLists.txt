cmake_minimum_required(VERSION 3.16)
project(eacpp CXX)

set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(SRC_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src")
file(GLOB_RECURSE SRC_FILES "${SRC_DIRECTORY}/*.cpp")
add_library(eacpp STATIC "${SRC_FILES}")

target_compile_features(eacpp PUBLIC cxx_std_20)

find_package(MPI REQUIRED)
add_subdirectory(extern/json)
add_subdirectory(extern/eigen)

target_link_libraries(eacpp Eigen3::Eigen MPI::MPI_CXX nlohmann_json::nlohmann_json)

target_include_directories(eacpp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_subdirectory(examples)
add_subdirectory(benchmarks)