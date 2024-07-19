# GoogleTestをダウンロード
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)

# 親プロジェクトのコンパイラ・リンカ設定を上書きするのを防ぐ（Windowsのみ）
if(WIN32)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif()

# googletestをターゲットとして使えるようにする
FetchContent_MakeAvailable(googletest)

add_subdirectory(Crossovers)
add_subdirectory(Decompositions)
add_subdirectory(Mutations)
add_subdirectory(Problems)
add_subdirectory(Rng)
add_subdirectory(Samplings)
add_subdirectory(Selections)
add_subdirectory(Utils)