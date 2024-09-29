#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "Problems/IBenchmark.h"

namespace Eacpp {

class BenchmarkReflection {
   public:
    using CreateFunction = std::unique_ptr<IBenchmark> (*)();

    static std::unordered_map<std::string, CreateFunction>& GetRegistry();
    static std::unique_ptr<IBenchmark> Create(const std::string& type);
    static void Register(const std::string& type, CreateFunction func);
};

template <typename T>
class BenchmarkReflectionRegister {
   public:
    BenchmarkReflectionRegister(const std::string& type) {
        BenchmarkReflection::Register(type, &Create);
    }
    static std::unique_ptr<IBenchmark> Create() {
        return std::make_unique<T>();
    }
};

#define REGISTER_BENCHMARK_REFLECTION(type)                         \
    namespace type##Reflection {                                    \
        static BenchmarkReflectionRegister<type> reflection(#type); \
    }

}  // namespace Eacpp