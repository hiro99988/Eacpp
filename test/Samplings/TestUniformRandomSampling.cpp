#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <eigen3/Eigen/Core>
#include <vector>

#include "Rng/MockRng.h"
#include "Samplings/UniformRandomSampling.h"

using ::testing::Return;

namespace Eacpp::Test {

TEST(UniformRandomSamplingTest, Sample) {
    UniformRandomSampling sampling(0.0, 1.0);
    std::vector<Eigen::ArrayXd> actual = sampling.Sample(5, 3);
    ASSERT_TRUE(
        std::all_of(actual.begin(), actual.end(), [&](Eigen::ArrayXd x) { return (x >= 0.0).all() && (x <= 1.0).all(); }));
}

}  // namespace Eacpp::Test
