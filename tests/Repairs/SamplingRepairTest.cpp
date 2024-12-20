#include <gtest/gtest.h>

#include "Individual.h"
#include "Repairs/SamplingRepair.h"
#include "Samplings/MockSampling.h"

namespace Eacpp {

TEST(SamplingRepairTest, Repair) {
    Individuali expected(Eigen::ArrayXi::LinSpaced(3, 1, 3));
    Individuali Individual(Eigen::ArrayXi::Zero(3));

    auto mockSampling = std::make_shared<MockSampling<int>>();
    EXPECT_CALL(*mockSampling, Sample(1, 3))
        .WillOnce(testing::Return(std::vector<Individuali>{expected}));

    SamplingRepair<int> repair(mockSampling);

    repair.Repair(Individual);

    EXPECT_EQ(expected, Individual);
}

}  // namespace Eacpp