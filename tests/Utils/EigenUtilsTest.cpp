#include <gtest/gtest.h>

#include <eigen3/Eigen/Core>

#include "Utils/EigenUtils.h"

namespace Eacpp {

TEST(EigenUtilsTest, AreEqual) {
    Eigen::ArrayXi lhs;
    Eigen::ArrayXi rhs;

    lhs.resize(3);
    rhs.resize(2);
    EXPECT_FALSE(AreEqual(lhs, rhs));

    lhs.resize(3);
    rhs.resize(3);
    lhs << 0, 1, 2;
    rhs << 0, 0, 0;
    EXPECT_FALSE(AreEqual(lhs, rhs));

    lhs << 0, 1, 2;
    rhs << 0, 1, 2;
    EXPECT_TRUE(AreEqual(lhs, rhs));
}

TEST(EigenUtilsTest, AreNotEqual) {
    Eigen::ArrayXi lhs;
    Eigen::ArrayXi rhs;

    lhs.resize(3);
    rhs.resize(2);
    EXPECT_TRUE(AreNotEqual(lhs, rhs));

    lhs.resize(3);
    rhs.resize(3);
    lhs << 0, 1, 2;
    rhs << 0, 0, 0;
    EXPECT_TRUE(AreNotEqual(lhs, rhs));

    lhs << 0, 1, 2;
    rhs << 0, 1, 2;
    EXPECT_FALSE(AreNotEqual(lhs, rhs));
}

TEST(EigenUtilsTest, AreCloseD) {
    Eigen::ArrayXd lhs;
    Eigen::ArrayXd rhs;
    double epsilon = 0.1;

    lhs.resize(3);
    rhs.resize(2);
    EXPECT_FALSE(AreClose(lhs, rhs, epsilon));

    lhs.resize(3);
    rhs.resize(3);
    lhs << 0.0, 1.0, 2.0;
    rhs << 0.0, 1.0, 2.2;
    EXPECT_FALSE(AreClose(lhs, rhs, epsilon));

    lhs << 0.0, 1.0, 2.0;
    rhs << 0.0, 1.0, 2.0;
    EXPECT_TRUE(AreClose(lhs, rhs, epsilon));
}

TEST(EigenUtilsTest, AreCloseF) {
    Eigen::ArrayXf lhs;
    Eigen::ArrayXf rhs;
    float epsilon = 0.1f;

    lhs.resize(3);
    rhs.resize(2);
    EXPECT_FALSE(AreClose(lhs, rhs, epsilon));

    lhs.resize(3);
    rhs.resize(3);
    lhs << 0.0f, 1.0f, 2.0f;
    rhs << 0.0f, 1.0f, 2.2f;
    EXPECT_FALSE(AreClose(lhs, rhs, epsilon));

    lhs << 0.0f, 1.0f, 2.0f;
    rhs << 0.0f, 1.0f, 2.0f;
    EXPECT_TRUE(AreClose(lhs, rhs, epsilon));
}

TEST(EigenUtilsTest, CalculateSquaredEuclideanDistance) {
    Eigen::ArrayXd lhs = Eigen::ArrayXd::LinSpaced(3, 0.0, 2.0);
    Eigen::ArrayXd rhs = Eigen::ArrayXd::LinSpaced(3, 1.0, 3.0);

    EXPECT_EQ(CalculateSquaredEuclideanDistance(lhs, rhs), 3.0);
}

TEST(EigenUtilsTest, CalculateEuclideanDistance) {
    Eigen::ArrayXd lhs = Eigen::ArrayXd::LinSpaced(3, 0.0, 2.0);
    Eigen::ArrayXd rhs = Eigen::ArrayXd::LinSpaced(3, 1.0, 3.0);

    EXPECT_DOUBLE_EQ(CalculateEuclideanDistance(lhs, rhs), std::sqrt(3.0));
}

}  // namespace Eacpp
