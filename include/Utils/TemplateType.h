#ifndef TmplateType_H
#define TmplateType_H

#include <concepts>

namespace Eacpp {

template <typename T>
concept Number = std::integral<T> || std::floating_point<T>;

}

#endif