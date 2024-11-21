#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <eigen3/Eigen/Core>
#include <memory>

#include "Crossovers/BinomialCrossover.h"
#include "Individual.h"
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

    Individuald parent1(Eigen::ArrayXd::LinSpaced(5, 1, 5));
    Individuald parent2(Eigen::ArrayXd::LinSpaced(5, 6, 10));
    Individuald parent3(Eigen::ArrayXd::LinSpaced(5, 11, 15));
    std::vector<Individuald> parents = {parent1, parent2, parent3};
    Individuald expected(5);
    expected.solution << -1.5, -0.5, 0.5, 1.5, 2.5;

    auto actual = binomialCrossover.Cross(parents);
    ASSERT_TRUE(actual == expected);
}

}  // namespace Eacpp::Test