#ifndef TmplateType_H
#define TmplateType_H

#include <concepts>
#include <cstdint>

namespace Eacpp {

template <typename T>
concept Number = std::integral<T> || std::floating_point<T>;

using SeedType = std::uint_fast32_t;

}  // namespace Eacpp

#endif