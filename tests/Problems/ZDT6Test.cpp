#define _TEST_

#include <gtest/gtest.h>

#include <cmath>
#include <eigen3/Eigen/Core>
#include <numbers>

#include "Problems/ZDT6.h"

namespace Eacpp {

class ZDT6Test : public ::testing::Test {
   protected:
    void SetUp() override {}
    void TearDown() override {}

    ZDT6 zdt6{};

    double F1(double x1) {
        return zdt6.F1(x1);
    }
    double G(const Eigen::ArrayXd& X) {
        return zdt6.G(X);
    }
    double F2(double f1, double g) {
        return zdt6.F2(f1, g);
    }
};

}  // namespace Eacpp

namespace Eacpp::Test {

TEST_F(ZDT6Test, F1) {
    ASSERT_DOUBLE_EQ(F1(1.0), 1.0 - std::exp(-4.0) * std::pow(std::sin(6.0 * std::numbers::pi), 6.0));
}

TEST_F(ZDT6Test, G) {
    Eigen::ArrayXd X(2);
    X << 1.0, 2.0;
    ASSERT_DOUBLE_EQ(G(X), 1.0 + 9.0 * std::pow(1.0 / 3.0, 0.25));
}

TEST_F(ZDT6Test, F2) {
    ASSERT_DOUBLE_EQ(F2(6.0, 2.0), -8.0);
}

}  // namespace Eacpp::Test