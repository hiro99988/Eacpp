#include "Rng/Rng.h"

#include <hash_map>
#include <iostream>
#include <unordered_map>

#include "Utils/Utils.h"

namespace Eacpp {

int Rng::Integer(const int max) const { return Integer(0, max); }

int Rng::Integer(int min, int max) const {
    swapIfMaxLessThanMin(min, max);
    std::uniform_int_distribution<int> dist(min, max);
    return dist(mt);
}

std::vector<int> Rng::Integers(int min, int max, const int size, bool replace) const {
    swapIfMaxLessThanMin(min, max);
    std::vector<int> result;
    std::uniform_int_distribution<int> dist(min, max);
    if (replace) {
        for (int i = 0; i < size; i++) {
            result.push_back(dist(mt));
        }
        return result;
    } else {
        if (size > (max - min + 1)) {
            throw std::invalid_argument("Size must be less than or equal to the range of values");
        }
        using hash_map = std::unordered_map<int, int>;

        result.reserve(size);

        hash_map Map;

        for (size_t cnt = 0; cnt < size; ++cnt) {
            int val = dist(mt);
            hash_map::iterator itr = Map.find(val);

            int replaced_val;
            hash_map::iterator replaced_itr = Map.find(max);
            if (replaced_itr != Map.end())
                replaced_val = replaced_itr->second;
            else
                replaced_val = max;

            if (itr == Map.end()) {
                result.push_back(val);
                if (val != max) Map.insert(std::make_pair(val, replaced_val));
            } else {
                result.push_back(itr->second);
                itr->second = replaced_val;
            }

            --max;
        }

        return result;
    }
}

double Rng::Uniform(double min, double max) const {
    if (min > max) {
        std::swap(min, max);
    }
    std::uniform_real_distribution<double> dist(min, max);
    return dist(mt);
}

Eigen::ArrayXd Rng::Uniform(double min, double max, const int size) const {
    std::uniform_real_distribution<double> dist(min, max);
    Eigen::ArrayXd result = Eigen::ArrayXd::Zero(size).unaryExpr([&](double) { return dist(mt); });
    return result;
}

Eigen::ArrayXXd Rng::Uniform(double min, double max, const std::tuple<int, int> size) const {
    std::uniform_real_distribution<double> dist(min, max);
    Eigen::ArrayXXd result =
        Eigen::ArrayXXd::Zero(std::get<0>(size), std::get<1>(size)).unaryExpr([&](double) { return dist(mt); });
    return result;
}

double Rng::Random() const { return Uniform(0.0, 1.0); }
Eigen::ArrayXd Rng::Random(const int size) const { return Uniform(0.0, 1.0, size); }
Eigen::ArrayXXd Rng::Random(const std::tuple<int, int> size) const { return Uniform(0.0, 1.0, size); }

}  // namespace Eacpp
