#ifndef RngForTest_H
#define RngForTest_H

#include <eigen3/Eigen/Core>
#include <tuple>
#include <vector>

#include "Rng/IRng.h"

namespace Eacpp {

class RngForTest : public IRng {
   public:
    int intValue;
    double doubleValue;

    RngForTest(int value) : intValue(value), doubleValue(0.0){};
    RngForTest(double value) : intValue(0), doubleValue(value){};

    int Integer(const int max) const override { return intValue; }
    int Integer(int min, int max) const override { return intValue; }
    std::vector<int> Integers(int min, int max, const int size, bool replace) const override;

    double Uniform(double min, double max) const override { return doubleValue; }
    Eigen::ArrayXd Uniform(double min, double max, const int size) const override;
    Eigen::ArrayXXd Uniform(double min, double max, const std::tuple<int, int> size) const override;

    double Random() const override { return doubleValue; };
    Eigen::ArrayXd Random(const int size) const override;
    Eigen::ArrayXXd Random(const std::tuple<int, int> size) const override;
};
}  // namespace Eacpp

#endif