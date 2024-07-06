add_executable(TestUtils.out TestUtils.cpp)

target_link_libraries(TestUtils.out gtest_main Eacpp)

include(GoogleTest)
gtest_discover_tests(TestUtils.out)