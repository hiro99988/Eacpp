#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <eigen3/Eigen/Core>

#include "Rng/MockRng.h"
#include "Samplings/UniformRandomSampling.h"

using ::testing::Return;

namespace Eacpp::Test {

TEST(UniformRandomSamplingTest, Sample) {
    UniformRandomSampling sampling(0.0, 1.0);
    Eigen::ArrayXXd actual = sampling.Sample(5, 3);
    ASSERT_TRUE((actual >= 0.0).all() && (actual <= 1.0).all());
}

}  // namespace Eacpp::Test
