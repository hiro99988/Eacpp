#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <Eigen/Core>
#include <memory>

#include "Crossovers/OnePointCrossover.h"
#include "Individual.h"
#include "Rng/MockRng.h"

using ::testing::_;
using ::testing::Return;

namespace Eacpp::Test {

TEST(OnePointCrossoverTest, PerformCrossover) {
    std::shared_ptr<MockRng> mockRng = std::make_shared<MockRng>();
    EXPECT_CALL(*mockRng, Random()).WillRepeatedly(testing::Return(0.1));
    EXPECT_CALL(*mockRng, Integer(_, _))
        .Times(3)
        .WillOnce(Return(1))
        .WillOnce(Return(3))
        .WillOnce(Return(4));

    OnePointCrossover<int> onePointCrossover(1.0, mockRng);

    Individuali parent1(Eigen::ArrayXi::LinSpaced(5, 1, 5));
    Individuali parent2(Eigen::ArrayXi::LinSpaced(5, 6, 10));
    std::vector<Individuali> parents = {parent1, parent2};
    Individuali expected(5);

    auto actual = onePointCrossover.Cross(parents);
    expected.solution << 1, 7, 8, 9, 10;
    ASSERT_TRUE(actual == expected);

    actual = onePointCrossover.Cross(parents);
    expected.solution << 1, 2, 3, 9, 10;
    ASSERT_TRUE(actual == expected);

    actual = onePointCrossover.Cross(parents);
    expected.solution << 1, 2, 3, 4, 10;
    ASSERT_TRUE(actual == expected);
}

}  // namespace Eacpp::Test