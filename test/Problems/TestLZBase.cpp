#include <gtest/gtest.h>

#include "Problems/LZBase.h"

namespace Eacpp::Test {

class LZBaseTmp : public LZBase {
   public:
    LZBaseTmp(int objectiveNum, std::vector<std::array<double, 2>> variableBounds, int decisionNum)
        : LZBase(objectiveNum, variableBounds, decisionNum) {}

    Eigen::ArrayXd ComputeObjectiveSet(const Eigen::ArrayXd& solution) const override { return Eigen::ArrayXd::Zero(10); }
    double Beta(const Eigen::ArrayXd& solution, const int JIndex) const override { return 0.0; }
};

TEST(LZBaseTest, EvaluateConstraints) {
    int decisionNum = 10;
    LZBaseTmp lzBase1(2, {{0, 0.5}}, decisionNum);
    Eigen::ArrayXd solution = Eigen::ArrayXd::Constant(decisionNum, 0.2);
    int invalidIndex = 5;
    solution(invalidIndex) = 0.6;
    std::vector<bool> evaluation = lzBase1.EvaluateConstraints(solution);
    for (int i = 0; i < decisionNum; i++) {
        if (i == invalidIndex) {
            ASSERT_FALSE(evaluation[i]);
        } else {
            ASSERT_TRUE(evaluation[i]);
        }
    }

    LZBaseTmp lzBase2(2, {{0, 0.5}, {0, 0.5}}, decisionNum);
    solution = Eigen::ArrayXd::Constant(decisionNum, 0.2);
    invalidIndex = 1;
    solution(invalidIndex) = 0.6;
    evaluation = lzBase2.EvaluateConstraints(solution);
    for (int i = 0; i < decisionNum; i++) {
        if (i == invalidIndex) {
            ASSERT_FALSE(evaluation[i]);
        } else {
            ASSERT_TRUE(evaluation[i]);
        }
    }
}

TEST(LZBaseTest, IsFeasible) {
    int decisionNum = 10;
    LZBaseTmp lzBase1(2, {{0, 0.5}}, decisionNum);
    Eigen::ArrayXd solution = Eigen::ArrayXd::Constant(decisionNum, 0.2);
    ASSERT_TRUE(lzBase1.IsFeasible(solution));

    int invalidIndex = 5;
    solution(invalidIndex) = 0.6;
    ASSERT_FALSE(lzBase1.IsFeasible(solution));

    LZBaseTmp lzBase2(2, {{0, 0.5}, {0, 0.5}}, decisionNum);
    solution = Eigen::ArrayXd::Constant(decisionNum, 0.2);
    ASSERT_TRUE(lzBase2.IsFeasible(solution));

    invalidIndex = 1;
    solution(invalidIndex) = 0.6;
    ASSERT_FALSE(lzBase2.IsFeasible(solution));
}

}  // namespace Eacpp::Test