#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace Eacpp {

template <typename Interface>
class Reflection {
   public:
    using CreateFunction = std::unique_ptr<Interface> (*)();

    static std::unordered_map<std::string, CreateFunction>& GetRegistry() {
        static std::unordered_map<std::string, CreateFunction> registry;
        return registry;
    }

    static std::unique_ptr<Interface> Create(const std::string& type) {
        auto it = GetRegistry().find(type);
        if (it != GetRegistry().end()) {
            return it->second();
        }
        return nullptr;
    }

    static void Register(const std::string& type, CreateFunction func) {
        GetRegistry()[type] = func;
    }
};

template <typename Interface, typename T>
class ReflectionRegister {
   public:
    ReflectionRegister(const std::string& type) {
        Reflection<Interface>::Register(type, &Create);
    }
    static std::unique_ptr<Interface> Create() {
        return std::make_unique<T>();
    }
};

#define REGISTER_REFLECTION(interface, type, str)                    \
    namespace type##Reflection {                                     \
        static ReflectionRegister<interface, type> reflection(#str); \
    }

#define REGISTER_TEMPLATE_REFLECTION(interface, interfaceType, type, str)           \
    namespace interface##type##Reflection {                                         \
        static ReflectionRegister<interface<interfaceType>, type> reflection(#str); \
    }

}  // namespace Eacpp