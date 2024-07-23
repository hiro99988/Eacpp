#ifndef Utils_h
#define Utils_h

#include <eigen3/Eigen/Core>
#include <iostream>
#include <vector>

namespace Eacpp {

template <typename T>
void swapIfMaxLessThanMin(T &min, T &max) {
    if (max < min) {
        std::swap(min, max);
    }
}

inline std::vector<int> Rangei(const int start, const int end, const int step) {
    std::vector<int> tmp((end - start) / step + 1);
    for (int i = start, j = 0; i <= end; i += step, ++j) {
        tmp[j] = i;
    }
    return tmp;
}

inline Eigen::ArrayXi Rangeea(const int start, const int end, const int step) {
    int size = (end - start) / step + 1;
    Eigen::ArrayXi tmp = Eigen::ArrayXi::LinSpaced(size, start, start + step * (size - 1));
    return tmp;
}

namespace Utils {

template <typename T>
std::vector<std::vector<T>> ProductRecurse(const std::vector<T> &choices, const int repeat,
                                           const std::vector<std::vector<T>> &result) {
    if (repeat == 0) {
        return result;
    }

    std::vector<std::vector<T>> provisionalResult;  // 作成途中のデカルト積
    for (auto re : result) {
        for (auto c : choices) {
            std::vector<T> tmp = re;
            tmp.push_back(c);
            provisionalResult.push_back(tmp);
        }
    }

    return ProductRecurse(choices, repeat - 1, provisionalResult);
}

}  // namespace Utils

template <typename T>
std::vector<std::vector<T>> Product(const std::vector<T> &choices, const int repeat) {
    return Utils::ProductRecurse(choices, repeat, std::vector<std::vector<T>>(1, std::vector<T>()));
}

}  // namespace Eacpp

#endif