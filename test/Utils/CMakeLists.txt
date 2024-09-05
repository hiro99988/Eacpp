add_executable(TestUtils.out TestUtils.cpp)
add_executable(TestMpiUtils.out TestMpiUtils.cpp)

target_link_libraries(TestUtils.out gtest_main Eacpp)
target_link_libraries(TestMpiUtils.out gtest_main Eacpp MPI::MPI_CXX)

include(GoogleTest)
gtest_discover_tests(TestUtils.out)