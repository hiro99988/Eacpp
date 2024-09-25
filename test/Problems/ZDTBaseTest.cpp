#include <gtest/gtest.h>

#include <eigen3/Eigen/Core>

#include "Problems/ZDTBase.h"

namespace Eacpp::Test {

class ZDTBaseTest : public ::testing::Test {
   protected:
    void SetUp() override {}
    void TearDown() override {}

    class ZDTBaseTmp : public ZDTBase {
       public:
        ZDTBaseTmp(int decisionVariablesNum) : ZDTBase(decisionVariablesNum) {}
        double F1(double x1) const override { return 1.0; }
        double G(const Eigen::ArrayXd& X) const override { return 1.0; }
        double F2(double f1, double g) const override { return 2.0; }
    };

    ZDTBaseTmp zdtBaseTmp{1};
};

TEST_F(ZDTBaseTest, ComputeObjectiveSet) {
    Individuald individual(1);
    zdtBaseTmp.ComputeObjectiveSet(individual);
    Eigen::ArrayXd expected(2);
    expected << 1.0, 2.0;
    ASSERT_TRUE((individual.objectives == expected).all());
}

TEST_F(ZDTBaseTest, IsFeasible) {
    Individuald individual(1);
    individual.solution << 0.0;
    ASSERT_TRUE(zdtBaseTmp.IsFeasible(individual));
    individual.solution << -0.1;
    ASSERT_FALSE(zdtBaseTmp.IsFeasible(individual));
    individual.solution << 1.0;
    ASSERT_TRUE(zdtBaseTmp.IsFeasible(individual));
    individual.solution << 1.1;
    ASSERT_FALSE(zdtBaseTmp.IsFeasible(individual));
}

TEST_F(ZDTBaseTest, EvaluateConstraints) {
    Individuald individual(1);
    individual.solution << 0.0;
    std::vector<bool> expected = {true};
    ASSERT_EQ(zdtBaseTmp.EvaluateConstraints(individual), expected);
    individual.solution << -0.1;
    expected = {false};
    ASSERT_EQ(zdtBaseTmp.EvaluateConstraints(individual), expected);
    individual.solution << 1.0;
    expected = {true};
    ASSERT_EQ(zdtBaseTmp.EvaluateConstraints(individual), expected);
    individual.solution << 1.1;
    expected = {false};
    ASSERT_EQ(zdtBaseTmp.EvaluateConstraints(individual), expected);
}

}  // namespace Eacpp::Test