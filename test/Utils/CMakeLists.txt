file(GLOB_RECURSE TESTS Test*.cpp)

foreach(test ${TESTS})
    get_filename_component(target ${test} NAME_WE)
    add_executable(${target}.out ${test})
    target_link_libraries(${target}.out gtest_main Eacpp)
    include(GoogleTest)
    gtest_discover_tests(${target}.out)
endforeach()


file(GLOB_RECURSE MPI_TESTS MpiTest*.cpp)

foreach(test ${MPI_TESTS})
    get_filename_component(target ${test} NAME_WE)
    add_executable(${target}.out ${test})
    target_link_libraries(${target}.out gtest_main Eacpp MPI::MPI_CXX)
    include(GoogleTest)
    gtest_discover_tests(${target}.out)
endforeach()