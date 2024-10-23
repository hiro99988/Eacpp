#pragma once

#include <algorithm>
#include <eigen3/Eigen/Core>
#include <iostream>
#include <iterator>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <tuple>
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

inline std::vector<double> Ranged(const double start, const double end, const double step) {
    std::vector<double> tmp;
    tmp.reserve((end - start) / step + 1);
    for (double i = start; i <= end; i += step) {
        tmp.push_back(i);
    }
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
std::vector<T> TransformTo1d(const std::vector<std::vector<T>> &v) {
    std::vector<T> transformed;
    for (const auto &i : v) {
        transformed.insert(transformed.end(), i.begin(), i.end());
    }
    return transformed;
}

template <typename T>
std::vector<std::vector<T>> TransformTo2d(std::vector<T> &vec1d, int separation) {
    if (vec1d.size() % separation != 0) {
        throw std::invalid_argument("Vector size is not a multiple of separation");
    }
    int size = vec1d.size() / separation;
    std::vector<std::vector<T>> transformed(size);
    for (int i = 0; i < size; i++) {
        transformed[i] = std::vector<T>(vec1d.begin() + i * separation, vec1d.begin() + (i + 1) * separation);
    }
    return transformed;
}

template <typename T>
std::vector<Eigen::ArrayX<T>> TransformToEigenArrayX2d(std::vector<T> &vec1d, int separation) {
    if (vec1d.size() % separation != 0) {
        throw std::invalid_argument("Vector size is not a multiple of separation");
    }
    int size = vec1d.size() / separation;
    std::vector<Eigen::ArrayX<T>> transformed(size);
    for (int i = 0; i < size; i++) {
        transformed[i] = Eigen::Map<Eigen::ArrayX<T>>(vec1d.data() + i * separation, separation);
    }
    return transformed;
}

inline void CalculateMeanAndVariance(const std::vector<Eigen::ArrayXd> &data, double &mean, double &variance) {
    double sum = 0.0;
    double sumSquared = 0.0;
    int totalElements = 0;

    for (const auto &array : data) {
        for (auto &&i : array) {
            sum += i;
            sumSquared += i * i;
        }

        totalElements += array.size();
    }

    mean = sum / totalElements;
    variance = (sumSquared / totalElements) - (mean * mean);
}

template <std::ranges::range Range>
std::vector<size_t> ArgSort(const Range &range) {
    int size = std::ranges::distance(range);
    std::vector<size_t> indexes(size);
    std::iota(indexes.begin(), indexes.end(), 0);

    std::sort(indexes.begin(), indexes.end(), [&](size_t i, size_t j) {
        return *std::ranges::next(std::ranges::begin(range), i) < *std::ranges::next(std::ranges::begin(range), j);
    });

    return indexes;
}

template <typename T>
std::vector<T> LinSpace(T start, T end, int division) {
    std::vector<T> result;
    result.reserve(division);
    T val = start;
    T dif = (end - start) / static_cast<T>(division - 1);
    for (int i = 0; i < division; ++i) {
        result.push_back(val);
        val += dif;
    }

    return result;
};

}  // namespace Eacpp
