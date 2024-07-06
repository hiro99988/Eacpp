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