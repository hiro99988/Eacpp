#define _TEST_

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>
#include <memory>
#include <tuple>
#include <vector>

#include "Individual.h"
#include "Mutations/PolynomialMutation.h"
#include "Rng/MockRng.h"

using ::testing::Return;

namespace Eacpp {

// class PolynomialMutationTest : public ::testing::Test {
//    protected:
//     double distributionIndex;
//     std::vector<std::array<double, 3>> expectedSigma;
//     std::vector<std::vector<std::pair<double, double>>> variableBounds;

//     void SetUp() override {
//         distributionIndex = 20.0;
//         expectedSigma = {
//             {0.0, -1.0}, {0.2, std::pow(0.4, 1.0 / (distributionIndex + 1.0))
//             - 1.0}, {0.5, 0.0},  {0.8, 1.0 - std::pow(0.4, 1.0 /
//             (distributionIndex + 1.0))}, {1.0, 1.0},
//         };
//         variableBounds = {{{0.0, 1.0}}, {{0.0, 1.0}, {0.0, 2.0}}};
//     }

//     void PerformMutation(PolynomialMutation& mutation, int index,
//     Individuald& individual, double sigma) {
//         mutation.PerformMutation(index, individual, sigma);
//     }
// };

// }  // namespace Eacpp

// namespace Eacpp::Test {

// TEST_F(PolynomialMutationTest, PerformMutation) {
//     for (auto&& bound : variableBounds) {
//         std::shared_ptr<MockRng> rng = std::make_shared<MockRng>();
//         PolynomialMutation mutation(1.0, distributionIndex, bound, rng);
//         Individuald individual(3);
//         Individuald expected(3);

//         for (auto&& e : expectedSigma) {
//             individual.solution << 1.0, 1.0, 1.0;
//             double sigma = 1.0;
//             int index = 0;
//             PerformMutation(mutation, index, individual, sigma);
//             EXPECT_DOUBLE_EQ(individual.solution(index), 2.0);

//             index = 2;
//             PerformMutation(mutation, index, individual, sigma);
//             if (bound.size() == 1) {
//                 EXPECT_DOUBLE_EQ(individual.solution(index), 2.0);
//             } else {
//                 EXPECT_DOUBLE_EQ(individual.solution(index), 3.0);
//             }
//         }
//     }
// }

// TEST_F(PolynomialMutationTest, Mutate) {
//     std::shared_ptr<MockRng> rng = std::make_shared<MockRng>();
//     Eacpp::PolynomialMutation mutation(0.5, distributionIndex,
//     variableBounds[0], rng); Individuald
//     individual(Eigen::ArrayXd::Zero(10)); Individuald copy = individual;

//     EXPECT_CALL(*rng, Random()).WillRepeatedly(Return(1.0)).Times(10);
//     mutation.Mutate(individual);
//     ASSERT_TRUE(individual == copy);
// }

}  // namespace Eacpp
