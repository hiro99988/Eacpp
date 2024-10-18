#define _TEST_

#include <gtest/gtest.h>

#include <cmath>
#include <eigen3/Eigen/Core>

#include "Problems/ZDT1.h"

namespace Eacpp {

class ZDT1Test : public ::testing::Test {
   protected:
    void SetUp() override {}
    void TearDown() override {}

    ZDT1 zdt1{};

    double F1(double x1) {
        return zdt1.F1(x1);
    }
    double G(const Eigen::ArrayXd& X) {
        return zdt1.G(X);
    }
    double H(double f1, double g) {
        return zdt1.H(f1, g);
    }
};

}  // namespace Eacpp

namespace Eacpp::Test {

TEST_F(ZDT1Test, F1) {
    ASSERT_DOUBLE_EQ(F1(1.0), 1.0);
}

TEST_F(ZDT1Test, G) {
    Eigen::ArrayXd X(2);
    X << 1.0, 2.0;
    ASSERT_DOUBLE_EQ(G(X), 1.0 + 9.0 * (1.0 / 29.0 + 2.0 / 29.0));
}

TEST_F(ZDT1Test, H) {
    ASSERT_DOUBLE_EQ(H(3.0, 2.0), 1.0 - std::sqrt(3.0 / 2.0));
}

}  // namespace Eacpp::Test