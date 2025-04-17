#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <Eigen/Core>
#include <algorithm>
#include <memory>
#include <tuple>
#include <vector>

#include "Individual.h"
#include "Rng/MockRng.h"
#include "Samplings/RealRandomSampling.h"
#include "Utils/EigenUtils.h"

using ::testing::_;
using ::testing::Return;

namespace Eacpp::Test {

TEST(UniformRandomSamplingTest, Constructor) {
    double min = 0.0;
    double max = 1.0;
    RealRandomSampling sampling(min, max);
    EXPECT_EQ(sampling.VariableBounds().size(), 1);
    EXPECT_EQ(sampling.VariableBounds()[0].first, min);
    EXPECT_EQ(sampling.VariableBounds()[0].second, max);

    std::pair<double, double> variableBound = {min, max};
    sampling = RealRandomSampling(variableBound);
    EXPECT_EQ(sampling.VariableBounds().size(), 1);
    EXPECT_EQ(sampling.VariableBounds()[0].first, min);
    EXPECT_EQ(sampling.VariableBounds()[0].second, max);

    std::vector<std::pair<double, double>> variableBounds = {{min, max}};
    sampling = RealRandomSampling(variableBounds);
    EXPECT_EQ(sampling.VariableBounds().size(), 1);
    EXPECT_EQ(sampling.VariableBounds()[0].first, min);
    EXPECT_EQ(sampling.VariableBounds()[0].second, max);
}

TEST(UniformRandomSamplingTest, Sample) {
    double min = 0.0;
    double max = 1.0;
    auto rng = std::make_shared<MockRng>();
    RealRandomSampling sampling(min, max, rng);
    int sampleNum = 1;
    int variableNum = 2;
    std::pair<int, int> size = {sampleNum, variableNum};
    Eigen::ArrayXd sample = Eigen::ArrayXd::LinSpaced(2, 0, 1);
    std::vector<Eigen::ArrayXd> samples = {sample};
    EXPECT_CALL(*rng, Uniform(min, max, size)).WillOnce(Return(samples));

    auto actual = sampling.Sample(sampleNum, variableNum);
    EXPECT_EQ(actual.size(), sampleNum);
    EXPECT_TRUE(AreEqual(actual[0].solution, sample));

    std::vector<std::pair<double, double>> variableBounds = {{0.0, 1.0},
                                                             {1.0, 2.0}};
    sampling = RealRandomSampling(variableBounds, rng);
    sampleNum = 2;
    variableNum = 3;
    sample = Eigen::ArrayXd(3);
    sample << variableBounds[0].first, variableBounds[1].first,
        variableBounds[1].first;
    EXPECT_CALL(*rng,
                Uniform(variableBounds[0].first, variableBounds[0].second))
        .Times(sampleNum)
        .WillRepeatedly(Return(0.0));
    EXPECT_CALL(*rng,
                Uniform(variableBounds[1].first, variableBounds[1].second))
        .Times((variableNum - 1) * sampleNum)
        .WillRepeatedly(Return(1.0));

    actual = sampling.Sample(sampleNum, variableNum);
    EXPECT_EQ(actual.size(), sampleNum);
    for (int i = 0; i < actual.size(); ++i) {
        EXPECT_TRUE(AreEqual(actual[i].solution, sample));
    }
}

}  // namespace Eacpp::Test
