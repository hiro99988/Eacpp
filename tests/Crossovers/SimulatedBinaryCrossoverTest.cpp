#define _TEST_

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>
#include <memory>
#include <vector>

#include "Crossovers/SimulatedBinaryCrossover.h"
#include "Individual.h"
#include "Rng/MockRng.h"
#include "Utils/EigenUtils.h"

using ::testing::Return;

namespace Eacpp {

class SimulatedBinaryCrossoverTest : public ::testing::Test {
   protected:
    std::shared_ptr<MockRng> rng;
    std::vector<std::pair<double, double>> variableBounds;
    SimulatedBinaryCrossover crossover{0.0, {}};
    double alpha;
    double distributionIndex;

    void SetUp() override {
        rng = std::make_shared<MockRng>();
        variableBounds = {{0.0, 1.0}, {1.0, 2.0}, {2.0, 3.0}};
        alpha = 1.0 / 0.4;
        distributionIndex = 3.0;
        crossover = SimulatedBinaryCrossover(0.5, distributionIndex,
                                             variableBounds, rng);
    }

    Individuald PerformCrossover(const SimulatedBinaryCrossover& crossover,
                                 const std::vector<Individuald>& parents) {
        return crossover.performCrossover(parents);
    }

    double Betaq(const SimulatedBinaryCrossover& crossover, double alpha) {
        return crossover.Betaq(alpha);
    }

    double Alpha(const SimulatedBinaryCrossover& crossover, double beta) {
        return crossover.Alpha(beta);
    }

    double Beta1(const SimulatedBinaryCrossover& crossover, double x1,
                 double x2, double lowerBound) {
        return crossover.Beta1(x1, x2, lowerBound);
    }

    double Beta2(const SimulatedBinaryCrossover& crossover, double x1,
                 double x2, double upperBound) {
        return crossover.Beta2(x1, x2, upperBound);
    }
};

}  // namespace Eacpp

namespace Eacpp::Test {

TEST_F(SimulatedBinaryCrossoverTest, PerformCrossover) {
    Eigen::ArrayXd parent1(4);
    Eigen::ArrayXd parent2(4);
    parent1 << 0.35, 1.92, 2.33, 2.55;
    parent2 << 0.72, 1.28, 2.81, 2.11;
    std::vector<Individuald> parents = {Individuald(parent1),
                                        Individuald(parent2)};

    EXPECT_CALL(*rng, Random())
        .WillOnce(Return(1.0))
        .WillOnce(Return(0.0))
        .WillOnce(Return(1.0))
        .WillOnce(Return(1.0))
        .WillOnce(Return(1.0))
        .WillOnce(Return(0.0))
        .WillOnce(Return(1.0))
        .WillOnce(Return(1.0));
    Individuald child = PerformCrossover(crossover, parents);
    for (int i = 0; i < child.solution.size(); i++) {
        if (i % 2 == 0) {
            EXPECT_DOUBLE_EQ(child.solution[i], parent1[i]);
        } else {
            EXPECT_DOUBLE_EQ(child.solution[i], parent2[i]);
        }
    }

    EXPECT_CALL(*rng, Random())
        .WillOnce(Return(0.0))
        .WillOnce(Return(0.0))
        .WillOnce(Return(0.2))
        .WillOnce(Return(0.0))
        .WillOnce(Return(1.0))
        .WillOnce(Return(0.8))
        .WillOnce(Return(0.0))
        .WillOnce(Return(0.0))
        .WillOnce(Return(0.2))
        .WillOnce(Return(0.0))
        .WillOnce(Return(1.0))
        .WillOnce(Return(0.8));

    child = PerformCrossover(crossover, parents);

    for (int i = 0; i < child.solution.size(); i++) {
        if (i % 2 == 0) {
            double beta = Beta1(crossover, parent1[i], parent2[i],
                                variableBounds[i].first);
            double alpha = Alpha(crossover, beta);
            double betaq =
                std::pow(0.2 * alpha, 1.0 / (distributionIndex + 1.0));
            EXPECT_DOUBLE_EQ(child.solution[i],
                             0.5 * ((1.0 + betaq) * parent1[i] +
                                    (1.0 - betaq) * parent2[i]));
        } else if (i % 2 == 1 && i < 3) {
            double beta = Beta2(crossover, parent1[i], parent2[i],
                                variableBounds[i].second);
            double alpha = Alpha(crossover, beta);
            double betaq = std::pow(1.0 / (2.0 - 0.8 * alpha),
                                    1.0 / (distributionIndex + 1.0));
            EXPECT_DOUBLE_EQ(child.solution[i],
                             0.5 * ((1.0 - betaq) * parent2[i] +
                                    (1.0 + betaq) * parent1[i]));
        } else {
            double beta = Beta2(crossover, parent1[i], parent2[i],
                                variableBounds.back().second);
            double alpha = Alpha(crossover, beta);
            double betaq = std::pow(1.0 / (2.0 - 0.8 * alpha),
                                    1.0 / (distributionIndex + 1.0));
            EXPECT_DOUBLE_EQ(child.solution[i],
                             0.5 * ((1.0 - betaq) * parent2[i] +
                                    (1.0 + betaq) * parent1[i]));
        }
    }
}

TEST_F(SimulatedBinaryCrossoverTest, Betaq) {
    EXPECT_CALL(*rng, Random()).WillOnce(Return(0.0));
    EXPECT_DOUBLE_EQ(Betaq(crossover, alpha), 0.0);

    EXPECT_CALL(*rng, Random()).WillOnce(Return(0.2));
    EXPECT_DOUBLE_EQ(Betaq(crossover, alpha),
                     std::pow(0.5, 1.0 / (distributionIndex + 1.0)));

    EXPECT_CALL(*rng, Random()).WillOnce(Return(0.4));
    EXPECT_DOUBLE_EQ(Betaq(crossover, alpha), 1.0);

    EXPECT_CALL(*rng, Random()).WillOnce(Return(0.5));
    EXPECT_DOUBLE_EQ(
        Betaq(crossover, alpha),
        std::pow(1.0 / (2.0 - 0.5 * alpha), 1.0 / (distributionIndex + 1.0)));

    EXPECT_CALL(*rng, Random()).WillOnce(Return(1.0));
    EXPECT_DOUBLE_EQ(Betaq(crossover, 1.21),
                     std::pow(1.0 / 0.79, 1.0 / (distributionIndex + 1.0)));
}

TEST_F(SimulatedBinaryCrossoverTest, Alpha) {
    EXPECT_DOUBLE_EQ(Alpha(crossover, 0.5),
                     2.0 - std::pow(0.5, -(distributionIndex + 1.0)));
    EXPECT_DOUBLE_EQ(Alpha(crossover, 1.0), 1.0);
}

TEST_F(SimulatedBinaryCrossoverTest, Beta1) {
    EXPECT_DOUBLE_EQ(Beta1(crossover, 0.34, 0.82, 0.0),
                     1.0 + 2.0 * 0.34 / 0.48);
}

TEST_F(SimulatedBinaryCrossoverTest, Beta2) {
    EXPECT_DOUBLE_EQ(Beta2(crossover, 0.34, 0.82, 1.0),
                     1.0 + 2.0 * 0.18 / 0.48);
}

}  // namespace Eacpp::Test