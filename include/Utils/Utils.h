#ifndef Utils_h
#define Utils_h

#include <iostream>

namespace Eacpp {

template <typename T>
void swapIfMaxLessThanMin(T& min, T& max) {
    if (max < min) {
        std::swap(min, max);
    }
}

}  // namespace Eacpp

#endif