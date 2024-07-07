#include <gmock/gmock.h>
#include <gtest/gtest.h>

// #include "Mutations/BitFlipMutation.h"
#include "Mutations/BitFlipMutation.h"
#include "Rng/MockRng.h"

using ::testing::Return;

TEST(BitFlipMutationTest, Mutate) {
    Eacpp::MockRng rng;
    Eacpp::BitFlipMutation mutation(0.5, &rng);
    Eigen::ArrayXi individual = Eigen::ArrayXi::Zero(10);
    Eigen::ArrayXi expected(10);
    expected << 1, 1, 1, 1, 1, 0, 0, 0, 0, 0;

    for (size_t i = 0; i < 5; ++i) {
        EXPECT_CALL(rng, Random()).WillOnce(Return(0.9)).RetiresOnSaturation();
    }
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_CALL(rng, Random()).WillOnce(Return(0.1)).RetiresOnSaturation();
    }

    mutation.Mutate(individual);

    ASSERT_TRUE((individual == expected).all());
}