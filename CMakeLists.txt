cmake_minimum_required(VERSION 3.16)
project(Eacpp CXX)

# GoogleTestをダウンロード
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)

# 親プロジェクトのコンパイラ・リンカ設定を上書きするのを防ぐ（Windowsのみ）
if(WIN32)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif()

# googletestをターゲットとして使えるようにする
FetchContent_MakeAvailable(googletest)

include(FetchContent)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

enable_testing()

set(SRC_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src")

file(GLOB_RECURSE SRC_FILES "${SRC_DIRECTORY}/*.cpp")

add_library(Eacpp STATIC "${SRC_FILES}")

target_compile_features(Eacpp PUBLIC cxx_std_20)

find_package(MPI REQUIRED)
target_link_libraries(Eacpp MPI::MPI_CXX gtest_main gmock_main nlohmann_json::nlohmann_json)

target_include_directories(Eacpp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_subdirectory(examples)
add_subdirectory(tests)
add_subdirectory(benchmarks)
