file(GLOB_RECURSE TESTS *.cpp)

foreach(test ${TESTS})
    get_filename_component(target ${test} NAME_WE)
    add_executable(${target}.out ${test})
    target_link_libraries(${target}.out Eacpp)
endforeach()