#include <gtest/gtest.h>

#include "Decompositions/DecompositionBase.h"

namespace Eacpp::Test {

class DecompositionBaseTest : public ::testing::Test {
   protected:
    DecompositionBaseTest() {}

    class DecompositionBaseTmp : public DecompositionBase {
       public:
        DecompositionBaseTmp(int objectivesNum) : DecompositionBase(objectivesNum) {}
        double ComputeObjective(const Eigen::ArrayXd& weight, const Eigen::ArrayXd& objectiveSet) const override { return 0.0; }
    };

    DecompositionBaseTmp decompositionBase{2};
};

TEST_F(DecompositionBaseTest, Constructor) {
    ASSERT_EQ(decompositionBase.IdealPoint().size(), 2);
    ASSERT_EQ(decompositionBase.IdealPoint()(0), std::numeric_limits<double>::max());
    ASSERT_EQ(decompositionBase.IdealPoint()(1), std::numeric_limits<double>::max());
}

TEST_F(DecompositionBaseTest, IdealPoint) {
    Eigen::ArrayXd expected = Eigen::ArrayXd::LinSpaced(2, 0, 1);
    decompositionBase.IdealPoint() = expected;
    ASSERT_TRUE((decompositionBase.IdealPoint() == expected).all());
}

TEST_F(DecompositionBaseTest, UpdateIdealPoint) {
    Eigen::ArrayXd objectiveSet(2);
    objectiveSet << 1.0, 2.0;
    decompositionBase.UpdateIdealPoint(objectiveSet);
    ASSERT_TRUE((decompositionBase.IdealPoint() == objectiveSet).all());
}

}  // namespace Eacpp::Test
