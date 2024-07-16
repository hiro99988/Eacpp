#include <gtest/gtest.h>

#include "Utils/Utils.h"

TEST(UtilsTest, SwapIfMaxLessThanMin) {
    int imin = 1;
    int imax = 2;
    Eacpp::swapIfMaxLessThanMin(imin, imax);
    ASSERT_EQ(1, imin);
    ASSERT_EQ(2, imax);

    double dmin = 2.1;
    double dmax = 1.1;
    Eacpp::swapIfMaxLessThanMin(dmin, dmax);
    ASSERT_EQ(1.1, dmin);
    ASSERT_EQ(2.1, dmax);
}

TEST(UtilsTest, Rangei) {
    int start = 1;
    int end = 10;
    for (int step = 1; step <= end; ++step) {
        std::vector<int> actual = Eacpp::Rangei(start, end, step);
        for (int i = 0, j = start; i < actual.size(); ++i, j += step) {
            ASSERT_EQ(j, actual[i]);
        }
    }
}

TEST(UtilsTest, Rangeea) {
    int start = 1;
    int end = 10;
    for (int step = 1; step <= end; ++step) {
        Eigen::ArrayXi actual = Eacpp::Rangeea(start, end, step);
        for (int i = 0, j = start; i < actual.size(); ++i, j += step) {
            ASSERT_EQ(j, actual(i));
        }
    }
}