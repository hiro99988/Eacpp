#include <gtest/gtest.h>

#include <cmath>
#include <eigen3/Eigen/Core>
#include <vector>

#include "Utils/Utils.h"

namespace Eacpp::Test {

TEST(UtilsTest, SwapIfMaxLessThanMin) {
    int imin = 1;
    int imax = 2;
    swapIfMaxLessThanMin(imin, imax);
    ASSERT_EQ(1, imin);
    ASSERT_EQ(2, imax);

    double dmin = 2.1;
    double dmax = 1.1;
    swapIfMaxLessThanMin(dmin, dmax);
    ASSERT_EQ(1.1, dmin);
    ASSERT_EQ(2.1, dmax);
}

TEST(UtilsTest, Rangei) {
    int start = 1;
    int end = 10;
    for (int step = 1; step <= end; ++step) {
        std::vector<int> actual = Rangei(start, end, step);
        for (int i = 0, j = start; i < actual.size(); ++i, j += step) {
            ASSERT_EQ(j, actual[i]);
        }
    }
}

TEST(UtilsTest, Rangeea) {
    int start = 1;
    int end = 10;
    for (int step = 1; step <= end; ++step) {
        Eigen::ArrayXi actual = Rangeea(start, end, step);
        for (int i = 0, j = start; i < actual.size(); ++i, j += step) {
            ASSERT_EQ(j, actual(i));
        }
    }
}

TEST(UtilsTest, Product) {
    std::vector<int> choices = {1, 2, 3};
    int repeat = 2;
    std::vector<std::vector<int>> actual = Product(choices, repeat);
    std::vector<std::vector<int>> expected = {{1, 1}, {1, 2}, {1, 3}, {2, 1}, {2, 2}, {2, 3}, {3, 1}, {3, 2}, {3, 3}};
    ASSERT_EQ(std::pow(choices.size(), repeat), actual.size());
    ASSERT_TRUE(expected == actual);
}

TEST(UtilsTest, Combination) {
    ASSERT_EQ(1, Combination(0, 0));
    ASSERT_EQ(1, Combination(1, 0));
    ASSERT_EQ(1, Combination(1, 1));
    ASSERT_EQ(1, Combination(2, 0));
    ASSERT_EQ(2, Combination(2, 1));
    ASSERT_EQ(1, Combination(2, 2));
    ASSERT_EQ(1, Combination(3, 0));
    ASSERT_EQ(3, Combination(3, 1));
    ASSERT_EQ(3, Combination(3, 2));
    ASSERT_EQ(1, Combination(3, 3));
}

TEST(UtilsTest, TransformTo1d) {
    std::vector<std::vector<int>> v2d = {{1}, {2, 3}, {4, 5, 6}};
    std::vector<int> actual = TransformTo1d(v2d);
    std::vector<int> expected = {1, 2, 3, 4, 5, 6};
    ASSERT_TRUE(expected == actual);
}

}  // namespace Eacpp::Test