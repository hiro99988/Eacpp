#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <eigen3/Eigen/Core>
#include <memory>

#include "Crossovers/BinomialCrossover.h"
#include "Rng/MockRng.h"

using ::testing::_;
using ::testing::Return;

namespace Eacpp::Test {

TEST(BinomialCrossoverTest, PerformCrossover) {
    std::shared_ptr<MockRng> mockRng = std::make_shared<MockRng>();
    int crossoverRate = 0.5;
    double scalingFactor = 0.5;
    BinomialCrossover binomialCrossover(crossoverRate, scalingFactor, mockRng);
    EXPECT_CALL(*mockRng, Integer(_)).Times(1).WillOnce(Return(1));
    EXPECT_CALL(*mockRng, Random()).WillRepeatedly(Return(0.0));

    Eigen::ArrayXd parent1(5);
    parent1 << 1, 2, 3, 4, 5;
    Eigen::ArrayXd parent2(5);
    parent2 << 6, 7, 8, 9, 10;
    Eigen::ArrayXd parent3(5);
    parent3 << 11, 12, 13, 14, 15;
    std::vector parents = {parent1, parent2, parent3};
    Eigen::ArrayXd expected(5);
    expected << -1.5, -0.5, 0.5, 1.5, 2.5;

    Eigen::ArrayXd actual = binomialCrossover.Cross(parents);
    ASSERT_TRUE((expected == actual).all());
}

}  // namespace Eacpp::Test