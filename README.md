# Eacpp

```console
$ git clone https://github.com/hiro99988/Eacpp.git
$ cd Eacpp
$ git submodule update --init --recursive
# デバッグビルド
$ cmake -S . -B ./out/build/debug -DCMAKE_BUILD_TYPE=Debug
$ cmake --build ./out/build/debug
# リリースビルド
$ cmake -S . -B ./out/build/release -DCMAKE_BUILD_TYPE=Release
$ cmake --build ./out/build/release
```