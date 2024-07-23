#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <eigen3/Eigen/Core>

#include "Crossovers/OnePointCrossover.h"
#include "Rng/MockRng.h"

using ::testing::_;
using ::testing::Return;

namespace Eacpp::Test {

TEST(OnePointCrossoverTest, PerformCrossover) {
    MockRng mockRng;
    OnePointCrossover<int> onePointCrossover(1.0, &mockRng);
    EXPECT_CALL(mockRng, Random()).WillRepeatedly(testing::Return(0.1));
    EXPECT_CALL(mockRng, Integer(_, _)).Times(3).WillOnce(Return(1)).WillOnce(Return(3)).WillOnce(Return(4));

    Eigen::ArrayXi parent1(5);
    parent1 << 1, 2, 3, 4, 5;
    Eigen::ArrayXi parent2(5);
    parent2 << 6, 7, 8, 9, 10;
    std::vector parents = {parent1, parent2};
    Eigen::ArrayXi expected(5);

    Eigen::ArrayXi actual = onePointCrossover.Cross(parents);
    expected << 1, 7, 8, 9, 10;
    ASSERT_TRUE((expected == actual).all());

    actual = onePointCrossover.Cross(parents);
    expected << 1, 2, 3, 9, 10;
    ASSERT_TRUE((expected == actual).all());

    actual = onePointCrossover.Cross(parents);
    expected << 1, 2, 3, 4, 10;
    ASSERT_TRUE((expected == actual).all());
}

}  // namespace Eacpp::Test