#define _TEST_

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <eigen3/Eigen/Core>
#include <memory>
#include <vector>

#include "Mutations/PolynomialMutation.h"
#include "Rng/MockRng.h"

using ::testing::Return;

namespace Eacpp {

class PolynomialMutationTest : public ::testing::Test {
   protected:
    double distributionIndex;
    std::vector<std::array<double, 3>> expectedSigma;
    std::vector<std::vector<std::array<double, 2>>> variableBounds;

    void SetUp() override {
        distributionIndex = 20.0;
        expectedSigma = {
            {0.0, 0.0, -1.0},
            {1.0, 0.0, 1.0 - std::pow(2.0, 1.0 / (distributionIndex + 1.0))},
            {0.0, 0.2, std::pow(0.4, 1.0 / (distributionIndex + 1.0)) - 1.0},
            {1.0, 0.2, 1.0 - std::pow(1.6, 1.0 / (distributionIndex + 1.0))},
            {0.0, 0.5, 0.0},
            {1.0, 0.5, 0.0},
            {0.0, 1.0, std::pow(2.0, 1.0 / (distributionIndex + 1.0)) - 1.0},
            {1.0, 1.0, 1.0},
        };
        variableBounds = {{{0.0, 1.0}}, {{0.0, 1.0}, {0.0, 2.0}}};
    }

    void PerformMutation(PolynomialMutation& mutation, int index, Eigen::ArrayXd& individual, double sigma) {
        mutation.PerformMutation(index, individual, sigma);
    }
    double Sigma(PolynomialMutation& mutation) const { return mutation.Sigma(); }
};

}  // namespace Eacpp

namespace Eacpp::Test {

TEST_F(PolynomialMutationTest, Sigma) {
    std::shared_ptr<MockRng> rng = std::make_shared<MockRng>();
    Eacpp::PolynomialMutation mutation(0.0, distributionIndex, {}, rng);

    for (auto&& e : expectedSigma) {
        EXPECT_CALL(*rng, Random()).WillOnce(Return(e[0])).WillOnce(Return(e[1]));
        EXPECT_DOUBLE_EQ(Sigma(mutation), e[2]);
    }
}

TEST_F(PolynomialMutationTest, PerformMutation) {
    for (auto&& bound : variableBounds) {
        std::shared_ptr<MockRng> rng = std::make_shared<MockRng>();
        PolynomialMutation mutation(1.0, distributionIndex, bound, rng);
        Eigen::ArrayXd individual(3);
        Eigen::ArrayXd expected(3);

        for (auto&& e : expectedSigma) {
            individual << 1.0, 1.0, 1.0;
            double sigma = 1.0;
            int index = 0;
            PerformMutation(mutation, index, individual, sigma);
            EXPECT_DOUBLE_EQ(individual[index], 2.0);

            index = 2;
            PerformMutation(mutation, index, individual, sigma);
            if (bound.size() == 1) {
                EXPECT_DOUBLE_EQ(individual[index], 2.0);
            } else {
                EXPECT_DOUBLE_EQ(individual[index], 3.0);
            }
        }
    }
}

TEST_F(PolynomialMutationTest, Mutate) {
    std::shared_ptr<MockRng> rng = std::make_shared<MockRng>();
    Eacpp::PolynomialMutation mutation(0.5, distributionIndex, variableBounds[0], rng);
    Eigen::ArrayXd individual = Eigen::ArrayXd::Zero(10);
    Eigen::ArrayXd copy = individual;

    EXPECT_CALL(*rng, Random()).WillRepeatedly(Return(1.0));
    mutation.Mutate(individual);
    ASSERT_TRUE((individual == copy).all());

    EXPECT_CALL(*rng, Random()).WillRepeatedly(Return(0.0));
    mutation.Mutate(individual);
    ASSERT_FALSE((individual == copy).any());
}

}  // namespace Eacpp::Test
