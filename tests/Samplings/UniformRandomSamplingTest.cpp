#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <eigen3/Eigen/Core>
#include <vector>

#include "Rng/MockRng.h"
#include "Samplings/UniformRandomSampling.h"

using ::testing::_;
using ::testing::Return;

namespace Eacpp::Test {

TEST(UniformRandomSamplingTest, ConstructorException) {
    EXPECT_THROW(UniformRandomSampling(1.0, 0.0), std::invalid_argument);
    EXPECT_NO_THROW(UniformRandomSampling(0.0, 1.0));
}

TEST(UniformRandomSamplingTest, Sample) {
    Eigen::ArrayXd sample = Eigen::ArrayXd::LinSpaced(3, 0.0, 1.0);
    std::vector<Eigen::ArrayXd> samples(5, sample);
    int sampleNum = 5;
    int variableNum = 3;
    std::pair<int, int> size = {sampleNum, variableNum};
    int min = 0.0;
    int max = 1.0;

    auto rng = std::make_shared<MockRng>();
    EXPECT_CALL(*rng, Uniform(min, max, size)).WillOnce(Return(samples));

    UniformRandomSampling sampling(min, max, rng);
    auto actual = sampling.Sample(sampleNum, variableNum);

    for (int i = 0; i < sampleNum; ++i) {
        EXPECT_TRUE((actual[i].solution == sample).all());
    }
}

}  // namespace Eacpp::Test
