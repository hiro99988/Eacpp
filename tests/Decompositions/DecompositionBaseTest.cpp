#include <gtest/gtest.h>

#include "Decompositions/DecompositionBase.h"

namespace Eacpp::Test {

class DecompositionBaseTest : public ::testing::Test {
   protected:
    DecompositionBaseTest() {}

    class DecompositionBaseTmp : public DecompositionBase {
       public:
        DecompositionBaseTmp() {}
        double ComputeObjective(const Eigen::ArrayXd& weight, const Eigen::ArrayXd& objectiveSet) const override {
            return 0.0;
        }
    };

    DecompositionBaseTmp decompositionBase;
};

TEST_F(DecompositionBaseTest, IdealPoint) {
    Eigen::ArrayXd expected = Eigen::ArrayXd::LinSpaced(2, 0, 1);
    decompositionBase.UpdateIdealPoint(expected);
    ASSERT_TRUE((decompositionBase.IdealPoint() == expected).all());
}

TEST_F(DecompositionBaseTest, InitializeIdealPoint) {
    int objectivesNum = 2;
    decompositionBase.InitializeIdealPoint(objectivesNum);
    Eigen::ArrayXd expected = Eigen::ArrayXd::Constant(objectivesNum, std::numeric_limits<double>::max());
    ASSERT_TRUE((decompositionBase.IdealPoint() == expected).all());
}

TEST_F(DecompositionBaseTest, UpdateIdealPoint) {
    Eigen::ArrayXd objectiveSet(2);
    objectiveSet << 1.0, 2.0;
    decompositionBase.UpdateIdealPoint(objectiveSet);
    ASSERT_TRUE((decompositionBase.IdealPoint() == objectiveSet).all());
}

}  // namespace Eacpp::Test
