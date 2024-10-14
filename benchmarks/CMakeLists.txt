# file(GLOB_RECURSE FILES *.cpp)

# foreach(test ${FILES})
# get_filename_component(target ${test} NAME_WE)
# add_executable(${target}.out ${test})
# target_link_libraries(${target}.out Eacpp)
# endforeach()

include(FetchContent)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

add_executable(MoeadBenchmark.out MoeadBenchmark.cpp)
target_link_libraries(MoeadBenchmark.out Eacpp nlohmann_json::nlohmann_json)

add_executable(ParetoFront.out ParetoFront.cpp)
target_link_libraries(ParetoFront.out Eacpp)