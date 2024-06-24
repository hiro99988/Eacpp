#ifndef Utils_h
#define Utils_h

#include <iostream>

namespace Eacpp {

void swapIfMaxLessThanMin(int& min, int& max) {
    if (max < min) {
        std::swap(min, max);
    }
}

}  // namespace Eacpp

#endif