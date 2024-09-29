file(GLOB_RECURSE FILES *.cpp)

foreach(test ${FILES})
    get_filename_component(target ${test} NAME_WE)
    add_executable(${target}.out ${test})
    target_link_libraries(${target}.out Eacpp)
endforeach()