#include <gtest/gtest.h>

#include <eigen3/Eigen/Core>

#include "Individual/Individual.h"

namespace Eacpp {

// TODO: テストを実装する
TEST(IndividualTest, EqualityOperator) {}

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
