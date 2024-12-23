#include "Rng/Rng.h"

#include <iostream>
#include <unordered_map>

#include "Utils/Utils.h"

namespace Eacpp {

int Rng::Integer(const int max) {
    return Integer(0, max);
}

int Rng::Integer(int min, int max) {
    swapIfMaxLessThanMin(min, max);
    std::uniform_int_distribution<int> dist(min, max);
    return dist(_mt);
}

std::vector<int> Rng::Integers(int min, int max, const int size, bool replace) {
    swapIfMaxLessThanMin(min, max);
    std::vector<int> result;
    result.reserve(size);
    if (replace) {
        std::uniform_int_distribution<int> dist(min, max);
        for (int i = 0; i < size; i++) {
            result.push_back(dist(_mt));
        }
        return result;
    } else {
        if (size > (max - min + 1)) {
            throw std::invalid_argument(
                "Size must be less than or equal to the range of values");
        }
        using hashMap = std::unordered_map<int, int>;

        hashMap map;

        for (size_t i = 0; i < size; ++i) {
            int val = std::uniform_int_distribution<int>(min, max)(_mt);
            hashMap::iterator itr = map.find(val);

            int replaced_val;
            hashMap::iterator replaced_itr = map.find(max);
            if (replaced_itr != map.end()) {
                replaced_val = replaced_itr->second;
            } else {
                replaced_val = max;
            }

            if (itr == map.end()) {
                result.push_back(val);
                if (val != max) map.insert(std::make_pair(val, replaced_val));
            } else {
                result.push_back(itr->second);
                itr->second = replaced_val;
            }

            --max;
        }

        return result;
    }
}

double Rng::Uniform(double min, double max) {
    swapIfMaxLessThanMin(min, max);
    std::uniform_real_distribution<double> dist(min, max);
    return dist(_mt);
}

Eigen::ArrayXd Rng::Uniform(double min, double max, const int size) {
    swapIfMaxLessThanMin(min, max);
    std::uniform_real_distribution<double> dist(min, max);
    Eigen::ArrayXd result =
        Eigen::ArrayXd::Zero(size).unaryExpr([&](double) { return dist(_mt); });
    return result;
}

std::vector<Eigen::ArrayXd> Rng::Uniform(double min, double max,
                                         const std::pair<int, int> size) {
    swapIfMaxLessThanMin(min, max);
    std::uniform_real_distribution<double> dist(min, max);
    std::vector<Eigen::ArrayXd> result;
    result.reserve(size.first);
    for (int i = 0; i < size.first; ++i) {
        result.push_back(
            Eigen::ArrayXd::Zero(size.second).unaryExpr([&](double) {
                return dist(_mt);
            }));
    }
    return result;
}

double Rng::Random() {
    return Uniform(0.0, 1.0);
}
Eigen::ArrayXd Rng::Random(const int size) {
    return Uniform(0.0, 1.0, size);
}
std::vector<Eigen::ArrayXd> Rng::Random(const std::pair<int, int> size) {
    return Uniform(0.0, 1.0, size);
}

std::vector<int> Rng::Choice(const std::vector<int>& vector, const int size,
                             const bool replace) {
    auto chosenIndex = Integers(0, vector.size() - 1, size, replace);
    std::vector<int> choice;
    choice.reserve(size);
    for (int i = 0; i < size; ++i) {
        choice.push_back(vector[chosenIndex[i]]);
    }
    return choice;
}

}  // namespace Eacpp
