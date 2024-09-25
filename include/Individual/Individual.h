#pragma once

#include <algorithm>
#include <eigen3/Eigen/Core>
#include <vector>

namespace Eacpp {

template <typename T>
struct Individual {
    Eigen::ArrayX<T> solution;
    Eigen::ArrayXd objectives;
    Eigen::ArrayXd weightVector;
    std::vector<std::reference_wrapper<Individual<T>>> neighborhood;

    Individual() {}

    explicit Individual(const int solutionSize) : solution(solutionSize) {}

    Individual(const int solutionSize, const int objectivesSize) : solution(solutionSize), objectives(objectivesSize) {}

    Individual(const Eigen::ArrayX<T>& solution) : solution(solution) {}

    Individual(const Eigen::ArrayX<T>& solution, const Eigen::ArrayXd& objectives)
        : solution(solution), objectives(objectives) {}

    Individual(const Eigen::ArrayX<T>& solution, const Eigen::ArrayXd& objectives, const Eigen::ArrayXd& weightVector)
        : solution(solution), objectives(objectives), weightVector(weightVector) {}

    Individual(const Eigen::ArrayX<T>& solution, const Eigen::ArrayXd& objectives, const Eigen::ArrayXd& weightVector,
               const std::vector<std::reference_wrapper<Individual<T>>>& neighborhood)
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
        return (solution == other.solution).all() && (objectives == other.objectives).all() &&
               (weightVector == other.weightVector).all() && neighborhood.size() == other.neighborhood.size() &&
               std::equal(neighborhood.begin(), neighborhood.end(), other.neighborhood.begin(),
                          [](const std::reference_wrapper<Individual<T>>& a, const std::reference_wrapper<Individual<T>>& b) {
                              return &a.get() == &b.get();
                          });
    }

    bool operator!=(const Individual& other) const { return !(*this == other); }

    void UpdateFrom(const Individual& other) {
        solution = other.solution;
        objectives = other.objectives;
    }

    double CalculateSquaredEuclideanDistance(const Individual& other) const {
        return (objectives - other.objectives).matrix().squaredNorm();
    }
};

using Individuali = Individual<int>;
using Individualf = Individual<float>;
using Individuald = Individual<double>;

}  // namespace Eacpp