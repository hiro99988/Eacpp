#include <gtest/gtest.h>

#include <Eigen/Core>
#include <algorithm>
#include <iostream>
#include <unordered_set>
#include <vector>

#include "Rng/Rng.h"
#include "Utils/Utils.h"

namespace Eacpp::Test {

TEST(RngTest, IntegerMax0) {
    Rng rng;
    int expected = 0;
    for (int i = 0; i < 10; i++) {
        int actual = rng.Integer(0);
        ASSERT_EQ(expected, actual);
    }
}

TEST(RngTest, IntegerMax2) {
    Rng rng;
    int expectedMin = 0;
    int expectedMax = 2;
    for (int i = 0; i < 10; i++) {
        int actual = rng.Integer(expectedMax);
        ASSERT_LE(expectedMin, actual);
        ASSERT_GE(expectedMax, actual);
    }
}

TEST(RngTest, IntegerMaxMinus2) {
    Rng rng;
    int expectedMin = -2;
    int expectedMax = 0;
    for (int i = 0; i < 10; i++) {
        int actual = rng.Integer(expectedMin);
        ASSERT_LE(expectedMin, actual);
        ASSERT_GE(expectedMax, actual);
    }
}

TEST(RngTest, IntegerMinEqualMax) {
    Rng rng;
    int expected = 0;
    for (int i = 0; i < 10; i++) {
        int actual = rng.Integer(expected, expected);
        ASSERT_EQ(expected, actual);
    }
}

TEST(RngTest, IntegerMinLessThanMax) {
    Rng rng;
    int expectedMin = 1;
    int expectedMax = 3;
    for (int i = 0; i < 10; i++) {
        int actual = rng.Integer(expectedMin, expectedMax);
        ASSERT_LE(expectedMin, actual);
        ASSERT_GE(expectedMax, actual);
    }
}

TEST(RngTest, IntegerMinGreaterThanMax) {
    Rng rng;
    int expectedMin = 1;
    int expectedMax = 3;
    for (int i = 0; i < 10; i++) {
        int actual = rng.Integer(expectedMax, expectedMin);
        ASSERT_LE(expectedMin, actual);
        ASSERT_GE(expectedMax, actual);
    }
}

TEST(RngTest, IntegersReplaceTrue) {
    Rng rng;
    int expectedMin = 0;
    int expectedMax = 2;
    int size = 10;
    std::vector<int> actual =
        rng.Integers(expectedMin, expectedMax, size, true);
    ASSERT_EQ(size, actual.size());
    for (int i = 0; i < size; i++) {
        ASSERT_LE(expectedMin, actual[i]);
        ASSERT_GE(expectedMax, actual[i]);
    }
}

TEST(RngTest, IntergersMinMaxLessThanSizeReplaceFalseException) {
    Rng rng;
    int expectedMin = 0;
    int expectedMax = 2;
    int size = 10;
    ASSERT_THROW(rng.Integers(expectedMin, expectedMax, size, false),
                 std::invalid_argument);
}

TEST(RngTest, IntegersReplaceFalse) {
    Rng rng;
    int expectedMin = 1;
    int expectedMax = 100;
    int size = 100;
    std::vector<int> actual =
        rng.Integers(expectedMin, expectedMax, size, false);
    ASSERT_EQ(size, actual.size());
    std::unordered_set<int> uniqueElements(actual.begin(), actual.end());
    ASSERT_EQ(size, uniqueElements.size());
}

TEST(RngTest, UniformMinGreaterThanMax) {
    Rng rng;
    double expectedMin = 0.0;
    double expectedMax = 1.0;
    for (int i = 0; i < 10; i++) {
        double actual = rng.Uniform(expectedMax, expectedMin);
        ASSERT_LE(expectedMin, actual);
        ASSERT_GE(expectedMax, actual);
    }
}

TEST(RngTest, UniformMinEqualMax) {
    Rng rng;
    double expected = 0.0;
    for (int i = 0; i < 10; i++) {
        double actual = rng.Uniform(expected, expected);
        ASSERT_EQ(expected, actual);
    }
}

TEST(RngTest, UniformMinLessThanMax) {
    Rng rng;
    double expectedMin = 0.0;
    double expectedMax = 1.0;
    for (int i = 0; i < 10; i++) {
        double actual = rng.Uniform(expectedMin, expectedMax);
        ASSERT_LE(expectedMin, actual);
        ASSERT_GE(expectedMax, actual);
    }
}

TEST(RngTest, UniformSize10) {
    Rng rng;
    double expectedMin = 0.0;
    double expectedMax = 1.0;
    int size = 10;
    Eigen::ArrayXd actual = rng.Uniform(expectedMin, expectedMax, size);
    ASSERT_EQ(size, actual.size());
    ASSERT_TRUE((actual >= expectedMin).all());
    ASSERT_TRUE((actual <= expectedMax).all());
}

TEST(RngTest, UniformSize10x10) {
    constexpr double min = 0.0;
    constexpr double max = 1.0;
    constexpr std::pair<int, int> size = {1000, 10};
    constexpr double expectedMean = (min + max) / 2;
    constexpr double expectedVariance = (max - min) * (max - min) / 12;

    Rng rng;
    std::vector<Eigen::ArrayXd> actual = rng.Uniform(min, max, size);

    double mean, variance;
    CalculateMeanAndVariance(actual, mean, variance);

    ASSERT_EQ(std::get<0>(size), actual.size());
    ASSERT_EQ(std::get<1>(size), actual[0].size());
    ASSERT_NEAR(expectedMean, mean, 0.01);
    ASSERT_NEAR(expectedVariance, variance, 0.01);
}

TEST(RngTest, ChoiceInt) {
    std::vector<int> vector = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    Rng rngWithDuplicates(0);
    auto expectedIndexWithDuplicates =
        rngWithDuplicates.Integers(0, vector.size() - 1, 2, true);
    Rng noduplicateRng(0);
    auto expectedNoduplicateIndex =
        noduplicateRng.Integers(0, vector.size() - 1, 5, false);

    int size = 2;
    Rng rng1(0);
    std::vector<int> actual = rng1.Choice(vector, size, true);
    ASSERT_EQ(size, actual.size());
    for (int i = 0; i < size; i++) {
        ASSERT_TRUE(vector[expectedIndexWithDuplicates[i]] == actual[i]);
    }

    Rng rng2(0);
    size = 5;
    actual = rng2.Choice(vector, size, false);
    ASSERT_EQ(size, actual.size());
    for (int i = 0; i < size; i++) {
        ASSERT_TRUE(vector[expectedNoduplicateIndex[i]] == actual[i]);
    }
}

}  // namespace Eacpp::Test