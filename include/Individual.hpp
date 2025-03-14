#pragma once

#include <iostream>
#include <vector>

#include "Utils/VectorUtils.hpp"

namespace Eacpp {

template <typename T, typename U>
struct Individual {
    std::vector<T> solution;
    std::vector<U> objectives;
    std::vector<double> weightVector;
    std::vector<int> neighborhood;

    Individual() {}

    explicit Individual(const int solutionSize) : solution(solutionSize) {}

    Individual(const int solutionSize, const int objectivesSize)
        : solution(solutionSize), objectives(objectivesSize) {}

    explicit Individual(const std::vector<T>& solution) : solution(solution) {}

    Individual(const std::vector<T>& solution, const std::vector<U>& objectives)
        : solution(solution), objectives(objectives) {}

    Individual(const std::vector<T>& solution, const std::vector<U>& objectives,
               const std::vector<double>& weightVector)
        : solution(solution),
          objectives(objectives),
          weightVector(weightVector) {}

    Individual(const std::vector<T>& solution, const std::vector<U>& objectives,
               const std::vector<double>& weightVector,
               const std::vector<int>& neighborhood)
        : solution(solution),
          objectives(objectives),
          weightVector(weightVector),
          neighborhood(neighborhood) {}

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
        return solution == other.solution && objectives == other.objectives &&
               weightVector == other.weightVector &&
               neighborhood == other.neighborhood;
    }

    bool operator!=(const Individual& other) const {
        return !(*this == other);
    }

    friend std::ostream& operator<<(std::ostream& os,
                                    const Individual& individual) {
        os << "Individual: { ";
        os << "solution: { " << Join(individual.solution, ", ") << " }, ";
        os << "objectives: { " << Join(individual.objectives, ", ") << " }, ";
        os << "weightVector: { " << Join(individual.weightVector, ", ")
           << " }, ";
        os << "neighborhood: { " << Join(individual.neighborhood, ", ") << " }";
        os << "}\n";
        return os;
    }

    void ReplaceSolutionAndObjective(const Individual& other) {
        solution = other.solution;
        objectives = other.objectives;
    }

    bool HasSameWeightVector(const Individual& other) const {
        return weightVector == other.weightVector;
    }
};

using Individualii = Individual<int, int>;
using Individualif = Individual<int, float>;
using Individualid = Individual<int, double>;
using Individualfi = Individual<float, int>;
using Individualff = Individual<float, float>;
using Individualfd = Individual<float, double>;
using Individualdi = Individual<double, int>;
using Individualdf = Individual<double, float>;
using Individualdd = Individual<double, double>;

}  // namespace Eacpp