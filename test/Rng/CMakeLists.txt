add_executable(TestRng.out TestRng.cpp)

target_link_libraries(TestRng.out gtest_main Eacpp)

include(GoogleTest)
gtest_discover_tests(TestRng.out)