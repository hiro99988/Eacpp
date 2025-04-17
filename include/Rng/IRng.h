#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <random>
#include <tuple>
#include <vector>

namespace Eacpp {

struct IRng {
    virtual ~IRng() {}

    virtual int Integer(const int max) = 0;
    virtual int Integer(int min, int max) = 0;
    virtual std::vector<int> Integers(int min, int max, const int size,
                                      bool replace) = 0;

    virtual double Uniform(double min, double max) = 0;
    virtual Eigen::ArrayXd Uniform(double min, double max, const int size) = 0;
    virtual std::vector<Eigen::ArrayXd> Uniform(
        double min, double max, const std::pair<int, int> size) = 0;

    virtual double Random() = 0;
    virtual Eigen::ArrayXd Random(const int size) = 0;
    virtual std::vector<Eigen::ArrayXd> Random(
        const std::pair<int, int> size) = 0;

    virtual std::vector<int> Choice(const std::vector<int>& array,
                                    const int size, const bool replace) = 0;
};

}  // namespace Eacpp
