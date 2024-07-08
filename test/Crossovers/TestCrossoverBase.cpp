#include <gtest/gtest.h>

#include "Crossovers/CrossoverBase.h"

namespace Eacpp::Test {

class CrossoverBaseTest : public ::testing::Test {
   protected:
    void SetUp() override {}
    void TearDown() override {}

    class CrossoverBaseTmp : public CrossoverBase<int> {
       public:
        CrossoverBaseTmp(int parentNum) : CrossoverBase<int>(parentNum) {}
        Eigen::ArrayX<int> performCrossover(const Eigen::ArrayXX<int>& parents) const override { return parents.row(0); }
    };

    CrossoverBaseTmp crossoverBase{2};
};

TEST_F(CrossoverBaseTest, GetParentNum) { EXPECT_EQ(crossoverBase.GetParentNum(), 2); }

TEST_F(CrossoverBaseTest, CrossException) {
    Eigen::ArrayXXi parents = Eigen::ArrayXXi::Zero(5, 3);
    EXPECT_THROW(crossoverBase.Cross(parents), std::invalid_argument);

    parents = Eigen::ArrayXXi::Zero(5, 1);
    EXPECT_THROW(crossoverBase.Cross(parents), std::invalid_argument);

    parents = Eigen::ArrayXXi::Zero(5, 2);
    EXPECT_NO_THROW(crossoverBase.Cross(parents));
}

}  // namespace Eacpp::Test