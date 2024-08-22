add_executable(TestMpMoead.out TestMpMoead.cpp)
target_link_libraries(TestMpMoead.out gtest_main gmock_main Eacpp)

# add_executable(TestMpMoeadMpi.out TestMpMoeadMpi.cpp)
# target_link_libraries(TestMpMoeadMpi.out gtest_main gmock_main Eacpp MPI::MPI_CXX)
include(GoogleTest)
gtest_discover_tests(TestMpMoead.out)

# gtest_discover_tests(TestMpMoeadMpi.out)