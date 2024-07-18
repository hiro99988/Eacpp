#ifndef MockRng_H
#define MockRng_H

#include <gmock/gmock.h>

#include <eigen3/Eigen/Core>
#include <tuple>
#include <vector>

#include "Rng/IRng.h"

namespace Eacpp {

class MockRng : public IRng {
   public:
    MOCK_METHOD1(Integer, int(const int));
    MOCK_METHOD2(Integer, int(int, int));
    MOCK_METHOD4(Integers, std::vector<int>(int, int, const int, bool));

    MOCK_METHOD2(Uniform, double(double, double));
    MOCK_METHOD3(Uniform, Eigen::ArrayXd(double, double, const int));
    MOCK_METHOD3(Uniform, Eigen::ArrayXXd(double, double, const std::tuple<int, int>));

    MOCK_METHOD0(Random, double());
    MOCK_METHOD1(Random, Eigen::ArrayXd(const int));
    MOCK_METHOD1(Random, Eigen::ArrayXXd(const std::tuple<int, int>));

    MOCK_METHOD3(Choice, Eigen::ArrayXXi(const Eigen::ArrayXXi&, const int, const bool));
    MOCK_METHOD3(Choice, Eigen::ArrayXXd(const Eigen::ArrayXXd&, const int, const bool));
};

}  // namespace Eacpp

#endif