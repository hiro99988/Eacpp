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

    Eigen::ArrayXXi parents(5, 2);
    parents << 1, 6, 2, 7, 3, 8, 4, 9, 5, 10;
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