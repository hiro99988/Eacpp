#include <gtest/gtest.h>

#include "Crossovers/CrossoverBase.h"

namespace Eacpp::Test {

class CrossoverBaseTest : public ::testing::Test {
   protected:
    void SetUp() override {}
    void TearDown() override {}

    class CrossoverBaseTmp : public CrossoverBase<int> {
       public:
        CrossoverBaseTmp(int parentNum, double crossoverRate) : CrossoverBase<int>(parentNum, crossoverRate) {}
        Eigen::ArrayXi performCrossover(const std::vector<Eigen::ArrayXi>& parents) const override { return parents[0]; }
    };

    CrossoverBaseTmp crossoverBase{2, 1.0};
};

TEST_F(CrossoverBaseTest, GetParentNum) { EXPECT_EQ(crossoverBase.GetParentNum(), 2); }

TEST_F(CrossoverBaseTest, CrossException) {
    Eigen::ArrayXi parent = Eigen::ArrayXi::Zero(5);
    std::vector parents = {parent};
    EXPECT_THROW(crossoverBase.Cross(parents), std::invalid_argument);

    parents.push_back(parent);
    EXPECT_NO_THROW(crossoverBase.Cross(parents));

    parents.push_back(parent);
    EXPECT_THROW(crossoverBase.Cross(parents), std::invalid_argument);
}

}  // namespace Eacpp::Test