#pragma once

#include <algorithm>
#include <eigen3/Eigen/Core>
#include <iostream>
#include <vector>

#include "Utils/EigenUtils.h"

namespace Eacpp {

template <typename T>
struct Individual {
    Eigen::ArrayX<T> solution;
    Eigen::ArrayXd objectives;
    Eigen::ArrayXd weightVector;
    std::vector<int> neighborhood;

    Individual() {}

    explicit Individual(const int solutionSize) : solution(solutionSize) {}

    Individual(const int solutionSize, const int objectivesSize) : solution(solutionSize), objectives(objectivesSize) {}

    Individual(const Eigen::ArrayX<T>& solution) : solution(solution) {}

    Individual(const Eigen::ArrayX<T>& solution, const Eigen::ArrayXd& objectives)
        : solution(solution), objectives(objectives) {}

    Individual(const Eigen::ArrayX<T>& solution, const Eigen::ArrayXd& objectives, const Eigen::ArrayXd& weightVector)
        : solution(solution), objectives(objectives), weightVector(weightVector) {}

    Individual(const Eigen::ArrayX<T>& solution, const Eigen::ArrayXd& objectives, const Eigen::ArrayXd& weightVector,
               const std::vector<int>& neighborhood)
        : solution(solution), objectives(objectives), weightVector(weightVector), neighborhood(neighborhood) {}

    Individual(const Individual& other)
        : solution(other.solution),
          objectives(other.objectives),
          weightVector(other.weightVector),
          neighborhood(other.neighborhood) {}

    Individual& operator=(const Individual& other) {
        if (this != &other) {
            solution = other.solution;
            objectives = other.objectives;
            weightVector = other.weightVector;
            neighborhood = other.neighborhood;
        }
        return *this;
    }

    bool operator==(const Individual& other) const {
        return AreEqual(solution, other.solution) && AreEqual(objectives, other.objectives) &&
               AreEqual(weightVector, other.weightVector) && neighborhood == other.neighborhood;
    }

    bool operator!=(const Individual& other) const {
        return !(*this == other);
    }

    friend std::ostream& operator<<(std::ostream& os, const Individual& individual) {
        os << "Individual: { ";
        os << "solution: { " << individual.solution.transpose() << " }, ";
        os << "objectives: { " << individual.objectives.transpose() << " }, ";
        os << "weightVector: { " << individual.weightVector.transpose() << " }, ";
        os << "neighborhood: { ";
        for (const auto& neighbor : individual.neighborhood) {
            os << neighbor << " ";
        }
        os << "} ";
        os << "}\n";
        return os;
    }

    void UpdateFrom(const Individual& other) {
        solution = other.solution;
        objectives = other.objectives;
    }

    bool IsWeightVectorEqual(const Individual& other) const {
        return AreEqual(weightVector, other.weightVector);
    }

    double CalculateSquaredEuclideanDistanceOfWeightVector(const Individual& other) const {
        return (weightVector - other.weightVector).matrix().squaredNorm();
    }
};

using Individuali = Individual<int>;
using Individualf = Individual<float>;
using Individuald = Individual<double>;

}  // namespace Eacpp