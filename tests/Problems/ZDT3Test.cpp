#define _TEST_

#include <gtest/gtest.h>

#include <cmath>
#include <eigen3/Eigen/Core>
#include <numbers>

#include "Problems/ZDT3.h"

namespace Eacpp {

class ZDT3Test : public ::testing::Test {
   protected:
    void SetUp() override {}
    void TearDown() override {}

    ZDT3 zdt3{};

    double F1(double x1) {
        return zdt3.F1(x1);
    }
    double G(const Eigen::ArrayXd& X) {
        return zdt3.G(X);
    }
    double F2(double f1, double g) {
        return zdt3.H(f1, g);
    }
};

}  // namespace Eacpp

namespace Eacpp::Test {

TEST_F(ZDT3Test, F1) {
    ASSERT_DOUBLE_EQ(F1(1.0), 1.0);
}

TEST_F(ZDT3Test, G) {
    Eigen::ArrayXd X(2);
    X << 1.0, 2.0;
    ASSERT_DOUBLE_EQ(G(X), 1.0 + 9.0 * (1.0 / 29.0 + 2.0 / 29.0));
}

TEST_F(ZDT3Test, F2) {
    ASSERT_DOUBLE_EQ(F2(3.0, 2.0), 1.0 - std::sqrt(3.0 / 2.0) - 1.5 * std::sin(30.0 * std::numbers::pi));
}

}  // namespace Eacpp::Test