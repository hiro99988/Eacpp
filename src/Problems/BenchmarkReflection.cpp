#include "Problems/BenchmarkReflection.h"

#include "Problems/IBenchmark.h"

namespace Eacpp {

// クラス生成関数を登録するマップ
std::unordered_map<std::string, BenchmarkReflection::CreateFunction>& BenchmarkReflection::GetRegistry() {
    static std::unordered_map<std::string, CreateFunction> registry;
    return registry;
}

std::unique_ptr<IBenchmark> BenchmarkReflection::Create(const std::string& type) {
    auto it = GetRegistry().find(type);
    if (it != GetRegistry().end()) {
        return it->second();
    }
    return nullptr;
}

void BenchmarkReflection::Register(const std::string& type, CreateFunction func) {
    GetRegistry()[type] = func;
}

}  // namespace Eacpp