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

    std::vector<std::vector<T>> provisionalResult;
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

inline long long Combination(int n, int r) {
    std::vector<std::vector<long long>> v(n + 1, std::vector<long long>(n + 1, 0));
    for (int i = 0; i < v.size(); i++) {
        v[i][0] = 1;
        v[i][i] = 1;
    }
    for (int j = 1; j < v.size(); j++) {
        for (int k = 1; k < j; k++) {
            v[j][k] = (v[j - 1][k - 1] + v[j - 1][k]);
        }
    }
    return v[n][r];
}

template <typename T>
std::vector<T> ConvertVectorFrom2dTo1d(const std::vector<std::vector<T>> &v) {
    std::vector<T> tmp;
    for (const auto &i : v) {
        tmp.insert(tmp.end(), i.begin(), i.end());
    }
    return tmp;
}

}  // namespace Eacpp

#endif