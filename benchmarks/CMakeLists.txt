# ベンチマーク出力ディレクトリパスの設定
add_definitions(-DBENCHMARK_OUTPUT_DIR="${CMAKE_SOURCE_DIR}/out/data/")

add_executable(ParallelBenchmark.out ParallelBenchmark.cpp)
target_link_libraries(ParallelBenchmark.out eacpp Eigen3::Eigen nlohmann_json::nlohmann_json MPI::MPI_CXX)