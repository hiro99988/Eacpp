add_executable(MpMoeadTest.out MpMoeadTest.cpp)
target_link_libraries(MpMoeadTest.out gtest_main gmock_main Eacpp)

add_executable(MpMoeadMpiTest.out MpMoeadMpiTest.cpp)
target_link_libraries(MpMoeadMpiTest.out gtest_main gmock_main Eacpp MPI::MPI_CXX)

include(GoogleTest)
gtest_discover_tests(MpMoeadTest.out)
gtest_discover_tests(MpMoeadMpiTest.out)