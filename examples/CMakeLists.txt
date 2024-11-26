# file(GLOB_RECURSE TESTS *.cpp)

# foreach(test ${TESTS})
# get_filename_component(target ${test} NAME_WE)
# add_executable(${target}.out ${test})
# target_link_libraries(${target}.out Eacpp)
# endforeach()

find_package(MPI REQUIRED)

add_executable(MoeadEx.out MoeadEx.cpp)
target_link_libraries(MoeadEx.out Eacpp)

add_executable(MpMoeadEx.out MpMoeadEx.cpp)
target_link_libraries(MpMoeadEx.out Eacpp MPI::MPI_CXX)

add_executable(NtMoeadEx.out NtMoeadEx.cpp)
target_link_libraries(NtMoeadEx.out Eacpp MPI::MPI_CXX nlohmann_json::nlohmann_json)

add_executable(Test.out Test.cpp)
target_link_libraries(Test.out Eacpp)

add_executable(GraphEx.out GraphEx.cpp)
target_link_libraries(GraphEx.out Eacpp)
