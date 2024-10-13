#include <gtest/gtest.h>

#include <cmath>
#include <eigen3/Eigen/Core>
#include <stdexcept>
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

TEST(UtilsTest, Ranged) {
    double start = 1.1;
    double end = 10.1;
    for (double step = 1.1; step <= end; step += 1.1) {
        std::vector<double> actual = Ranged(start, end, step);

        double expected = start;
        for (int i = 0; i < actual.size(); ++i, expected += step) {
            ASSERT_DOUBLE_EQ(expected, actual[i]);
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

TEST(UtilsTest, TransformTo2d) {
    std::vector<int> v1d = {1, 2, 3, 4, 5, 6};
    std::vector<std::vector<int>> expected = {{1, 2, 3}, {4, 5, 6}};
    ASSERT_NO_THROW(TransformTo2d(v1d, 3));
    std::vector<std::vector<int>> actual = TransformTo2d(v1d, 3);
    ASSERT_TRUE(expected == actual);

    ASSERT_THROW(TransformTo2d(v1d, 4), std::invalid_argument);
}

TEST(UtilsTest, TransformToEigenArrayX2d) {
    std::vector<int> v1d = {1, 2, 3, 4, 5, 6};
    std::vector<Eigen::ArrayXi> expected = {Eigen::ArrayXi::LinSpaced(3, 1, 3), Eigen::ArrayXi::LinSpaced(3, 4, 6)};
    ASSERT_NO_THROW(TransformToEigenArrayX2d(v1d, 3));
    std::vector<Eigen::ArrayXi> actual = TransformToEigenArrayX2d(v1d, 3);
    for (int i = 0; i < expected.size(); ++i) {
        ASSERT_TRUE((expected[i] == actual[i]).all());
    }

    ASSERT_THROW(TransformToEigenArrayX2d(v1d, 4), std::invalid_argument);
}

TEST(UtilsTest, CalculateMeanAndVariance) {
    std::vector<Eigen::ArrayXd> data = {Eigen::ArrayXd::LinSpaced(3, 1, 3), Eigen::ArrayXd::LinSpaced(3, 4, 6)};
    double mean = 0.0;
    double variance = 0.0;
    CalculateMeanAndVariance(data, mean, variance);
    ASSERT_DOUBLE_EQ(3.5, mean);
    ASSERT_DOUBLE_EQ(91.0 / 6.0 - 3.5 * 3.5, variance);
}

TEST(UtilsTest, ArgSort) {
    std::vector<int> array = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3};
    auto actual = ArgSort(array);
    std::vector<size_t> expected = {1, 3, 6, 0, 9, 2, 4, 8, 7, 5};
    ASSERT_TRUE(actual == expected);
}

}  // namespace Eacpp::Test