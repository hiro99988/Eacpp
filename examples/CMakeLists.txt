find_package(MPI REQUIRED)

add_executable(MoeadEx.out MoeadEx.cpp)
target_link_libraries(MoeadEx.out eacpp)

add_executable(MpMoeadEx.out MpMoeadEx.cpp)
target_link_libraries(MpMoeadEx.out eacpp MPI::MPI_CXX)