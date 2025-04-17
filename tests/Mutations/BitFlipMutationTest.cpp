#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <Eigen/Core>
#include <memory>

#include "Individual.h"
#include "Mutations/BitFlipMutation.h"
#include "Rng/MockRng.h"

using ::testing::Return;

namespace Eacpp::Test {

TEST(BitFlipMutationTest, Mutate) {
    std::shared_ptr<MockRng> rng = std::make_shared<MockRng>();
    Eacpp::BitFlipMutation mutation(0.5, rng);
    Individuali individual(Eigen::ArrayXi::Zero(10));
    Individuali expected(Eigen::ArrayXi::Ones(10));
    expected.solution << 1, 1, 1, 1, 1, 0, 0, 0, 0, 0;

    for (size_t i = 0; i < 5; ++i) {
        EXPECT_CALL(*rng, Random()).WillOnce(Return(0.9)).RetiresOnSaturation();
    }
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_CALL(*rng, Random()).WillOnce(Return(0.1)).RetiresOnSaturation();
    }

    mutation.Mutate(individual);

    ASSERT_TRUE(individual == expected);
}

}  // namespace Eacpp::Test