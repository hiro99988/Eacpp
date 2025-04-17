cmake_minimum_required(VERSION 3.16)
project(Eacpp CXX)

# 親プロジェクトのコンパイラ・リンカ設定を上書きするのを防ぐ（Windowsのみ）
if(WIN32)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif()

enable_testing()

set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(SRC_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src")
file(GLOB_RECURSE SRC_FILES "${SRC_DIRECTORY}/*.cpp")
add_library(Eacpp STATIC "${SRC_FILES}")

target_compile_features(Eacpp PUBLIC cxx_std_20)

find_package(MPI REQUIRED)
add_subdirectory(extern/pybind11)
add_subdirectory(extern/googletest)
add_subdirectory(extern/json)
add_subdirectory(extern/eigen)

target_link_libraries(Eacpp Eigen3::Eigen MPI::MPI_CXX gtest_main gmock_main nlohmann_json::nlohmann_json pybind11::embed)

target_include_directories(Eacpp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_subdirectory(examples)
add_subdirectory(benchmarks)

option(ENABLE_TESTS "Enable building of tests" OFF)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
