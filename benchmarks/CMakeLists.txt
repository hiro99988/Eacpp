# ベンチマーク出力ディレクトリパスの設定
add_definitions(-DBENCHMARK_OUTPUT_DIR="${CMAKE_SOURCE_DIR}/out/data/")

add_executable(MoeadBenchmark.out MoeadBenchmark.cpp)
target_link_libraries(MoeadBenchmark.out Eacpp Eigen3::Eigen nlohmann_json::nlohmann_json)

add_executable(ParetoFront.out ParetoFront.cpp)
target_link_libraries(ParetoFront.out Eacpp)

add_executable(ParallelMoeadBenchmark.out ParallelMoeadBenchmark.cpp)
target_link_libraries(ParallelMoeadBenchmark.out Eacpp Eigen3::Eigen nlohmann_json::nlohmann_json MPI::MPI_CXX)

add_executable(ParallelHpaBenchmark.out ParallelHpaBenchmark.cpp)
target_link_libraries(ParallelHpaBenchmark.out Eacpp Eigen3::Eigen nlohmann_json::nlohmann_json MPI::MPI_CXX pybind11::embed)