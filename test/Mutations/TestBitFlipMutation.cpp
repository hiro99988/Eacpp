#include <gmock/gmock.h>
#include <gtest/gtest.h>

// #include "Mutations/BitFlipMutation.h"
#include "Rng/MockRng.h"

using ::testing::Return;

TEST(BitFlipMutationTest, Mutate) {
    Eacpp::MockRng rng;

    EXPECT_CALL(rng, Random()).WillRepeatedly(Return(1.0));
    double actual = rng.Random();

    ASSERT_DOUBLE_EQ(1.0, actual);
}