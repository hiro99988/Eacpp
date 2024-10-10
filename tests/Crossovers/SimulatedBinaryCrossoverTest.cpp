#define _TEST_

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cmath>
#include <eigen3/Eigen/Core>
#include <memory>
#include <vector>

#include "Crossovers/SimulatedBinaryCrossover.h"
#include "Individual/Individual.h"
#include "Rng/MockRng.h"

using ::testing::Return;

namespace Eacpp {

class SimulatedBinaryCrossoverTest : public ::testing::Test {
   protected:
    std::shared_ptr<MockRng> rng;
    SimulatedBinaryCrossover crossover{0.5};

    void SetUp() override {
        rng = std::make_shared<MockRng>();
        crossover = SimulatedBinaryCrossover(0.5, 2.0, rng);
    }

    Individuald PerformCrossover(const SimulatedBinaryCrossover& crossover, const std::vector<Individuald>& parents) {
        return crossover.performCrossover(parents);
    }

    double Beta(const SimulatedBinaryCrossover& crossover) {
        return crossover.Beta();
    }
};

}  // namespace Eacpp

namespace Eacpp::Test {

TEST_F(SimulatedBinaryCrossoverTest, Beta) {
    EXPECT_CALL(*rng, Random()).WillOnce(Return(0.0));
    EXPECT_EQ(Beta(crossover), 0.0);

    EXPECT_CALL(*rng, Random()).WillOnce(Return(0.5));
    EXPECT_DOUBLE_EQ(Beta(crossover), std::pow(1.0, 1.0 / 3.0));

    EXPECT_CALL(*rng, Random()).WillOnce(Return(0.6));
    EXPECT_DOUBLE_EQ(Beta(crossover), std::pow(1.0 / 0.8, 1.0 / 3.0));

    EXPECT_CALL(*rng, Random()).WillOnce(Return(0.9));
    EXPECT_DOUBLE_EQ(Beta(crossover), std::pow(1.0 / 0.2, 1.0 / 3.0));
}

TEST_F(SimulatedBinaryCrossoverTest, PerformCrossover) {
    Eigen::ArrayXd parent1(1);
    Eigen::ArrayXd parent2(1);
    parent1 << 2.0;
    parent2 << 5.0;

    EXPECT_CALL(*rng, Random()).WillOnce(Return(0.0)).WillOnce(Return(0.5));

    std::vector<Individuald> parents = {Individuald(parent1), Individuald(parent2)};
    Individuald child = PerformCrossover(crossover, parents);

    double beta = std::pow(1.0, 1.0 / 3.0);
    double expected = 0.5 * ((1.0 + beta) * parent1(0) + (1.0 - beta) * parent2(0));
    EXPECT_EQ(child.solution(0), expected);
}

}  // namespace Eacpp::Test