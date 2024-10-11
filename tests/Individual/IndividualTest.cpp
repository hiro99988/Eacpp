#include <gtest/gtest.h>

#include <algorithm>
#include <eigen3/Eigen/Core>
#include <vector>

#include "Individual/Individual.h"
#include "Utils/EigenUtils.h"

namespace Eacpp {

TEST(IndividualTest, AssignmentOperator) {
    Individuali individual1;
    Individuali individual2;
    individual1.solution = Eigen::ArrayXi::LinSpaced(3, 0, 2);
    individual1.objectives = Eigen::ArrayXd::LinSpaced(3, 0.0, 2.0);
    individual1.weightVector = Eigen::ArrayXd::LinSpaced(3, 0.1, 0.3);
    individual1.neighborhood = {1, 2, 3};

    individual2 = individual1;

    EXPECT_EQ(individual1, individual2);
}

TEST(IndividualTest, EqualityOperator) {
    Eigen::ArrayXi solution = Eigen::ArrayXi::LinSpaced(3, 0, 2);
    Eigen::ArrayXd objectives = Eigen::ArrayXd::LinSpaced(3, 0.0, 2.0);
    Eigen::ArrayXd weightVector = Eigen::ArrayXd::LinSpaced(3, 0.1, 0.3);
    std::vector<int> neighborhood = {1, 2, 3};

    Individuali individual1(solution, objectives, weightVector, neighborhood);
    Individuali individual2(solution, objectives, weightVector, neighborhood);

    EXPECT_TRUE(individual1 == individual2);

    individual2.solution(0) = 100;
    EXPECT_FALSE(individual1 == individual2);

    individual2.solution = solution;
    individual2.objectives(0) = 100.0;
    EXPECT_FALSE(individual1 == individual2);

    individual2.objectives = objectives;
    individual2.weightVector(0) = 100.0;
    EXPECT_FALSE(individual1 == individual2);

    individual2.weightVector = weightVector;
    individual2.neighborhood = {1, 2};
    EXPECT_FALSE(individual1 == individual2);
}

TEST(IndividualTest, InequalityOperator) {
    Eigen::ArrayXi solution = Eigen::ArrayXi::LinSpaced(3, 0, 2);
    Eigen::ArrayXd objectives = Eigen::ArrayXd::LinSpaced(3, 0.0, 2.0);
    Eigen::ArrayXd weightVector = Eigen::ArrayXd::LinSpaced(3, 0.1, 0.3);
    std::vector<int> neighborhood = {1, 2, 3};

    Individuali individual1(solution, objectives, weightVector, neighborhood);
    Individuali individual2(solution, objectives, weightVector, neighborhood);

    EXPECT_FALSE(individual1 != individual2);

    individual2.solution(0) = 100;
    EXPECT_TRUE(individual1 != individual2);

    individual2.solution = solution;
    individual2.objectives(0) = 100.0;
    EXPECT_TRUE(individual1 != individual2);

    individual2.objectives = objectives;
    individual2.weightVector(0) = 100.0;
    EXPECT_TRUE(individual1 != individual2);

    individual2.weightVector = weightVector;
    individual2.neighborhood = {1, 2};
    EXPECT_TRUE(individual1 != individual2);
}

TEST(IndividualTest, OutputOperator) {
    Individuali individual;
    individual.solution = Eigen::ArrayXi::LinSpaced(3, 0, 2);
    individual.objectives = Eigen::ArrayXd::LinSpaced(3, 0.0, 2.0);
    individual.weightVector = Eigen::ArrayXd::LinSpaced(3, 0.1, 0.3);
    individual.neighborhood = {1, 2, 3};

    std::stringstream ss;
    ss << individual;

    std::string expected =
        "Individual: { solution: { 0 1 2 }, objectives: { 0 1 2 }, weightVector: { 0.1 0.2 0.3 }, neighborhood: { 1 2 3 } }\n";
    EXPECT_EQ(ss.str(), expected);
}

TEST(IndividualTest, UpdateFrom) {
    Individuali individual(Eigen::ArrayXi::Zero(3), Eigen::ArrayXd::Zero(3));
    Individuali newIndividual(Eigen::ArrayXi::Ones(3), Eigen::ArrayXd::Ones(3));
    individual.UpdateFrom(newIndividual);
    EXPECT_TRUE((individual.solution == newIndividual.solution).all());
    EXPECT_TRUE((individual.objectives == newIndividual.objectives).all());
}

TEST(IndividualTest, IsWeightVectorEqual) {
    Eigen::ArrayXd wv1 = Eigen::ArrayXd::LinSpaced(3, 0.1, 0.3);
    Eigen::ArrayXd wv2 = Eigen::ArrayXd::LinSpaced(3, 1.0, 3.0);
    Individuali individual1;
    Individuali individual2;
    individual1.weightVector = wv1;
    individual2.weightVector = wv2;

    EXPECT_FALSE(individual1.IsWeightVectorEqual(individual2));

    individual2.weightVector = wv1;

    EXPECT_TRUE(individual1.IsWeightVectorEqual(individual2));
}

TEST(IndividualTest, CalculateSquaredEuclideanDistanceOfWeightVector) {
    Individuali individual1;
    individual1.weightVector = Eigen::ArrayXd::LinSpaced(3, 0.1, 0.3);
    Individuali individual2;
    individual2.weightVector = Eigen::ArrayXd::LinSpaced(3, 1.0, 3.0);
    double expected = 11.34;

    double actual = individual1.CalculateSquaredEuclideanDistanceOfWeightVector(individual2);

    EXPECT_DOUBLE_EQ(expected, actual);
}

}  // namespace Eacpp
