#include "Algorithms/Moead.h"

#include <algorithm>
#include <eigen3/Eigen/Core>

namespace Eacpp {

template <Number DecisionVariableType>
void Moead<DecisionVariableType>::GenerateWeightVectors() {
    Eigen::ArrayXd takedSetTop = Eigen::ArrayXd::LinSpaced(H + 1, 0, H);
    Eigen::ArrayXXd product = Eigen::ArrayXXd::Zero(std::pow(H + 1, decisionVariableNum), decisionVariableNum);
    for (int i = 0; i < decisionVariableNum; i++) {
        product.col(i) = takedSetTop;
    }
    Eigen::ArrayXXd wvTop = product.rowwise().sum();
    weightVectors = wvTop.rowwise() / H;
}

// def generate_weight_vector(m, H):
//     taked_set_top = [i for i in range(H + 1)]
//     product = np.array(list(itertools.product(taked_set_top, repeat=m)))
//     wv_top = product[np.sum(product, axis=1) == H]
//     wv = wv_top / H
//     return wv

template <Number DecisionVariableType>
void Moead<DecisionVariableType>::GenerateNeighborhoods() {
    Eigen::ArrayXd euclideanDistances(populationSize, populationSize);
    for (int i = 0; i < populationSize; i++) {
        euclideanDistances.col(i) = (weightVectors.colwise() - weightVectors.col(i)).colwise().squaredNorm();
    }
    neighborhoodIndexes.resize(neighborNum, populationSize);
    for (int i = 0; i < populationSize; i++) {
        std::sort(euclideanDistances.col(i).begin(), euclideanDistances.col(i).end());
        neighborhoodIndexes.col(i) = euclideanDistances.col(i).head(neighborNum);
    }
}

template <Number DecisionVariableType>
void Moead<DecisionVariableType>::InitializePopulation() {
    solutions = sampling->Sample(populationSize, decisionVariableNum);
    objectiveSets.resize(objectiveNum, populationSize);
    for (int i = 0; i < populationSize; i++) {
        objectiveSets.col(i) = problem->ComputeObjectiveSet(solutions.row(i));
    }
}

template <Number DecisionVariableType>
void Moead<DecisionVariableType>::InitializeIdealPoint() {
    idealPoint = objectiveSets.rowwise().minCoeff();
}

template <Number DecisionVariableType>
Eigen::ArrayX<DecisionVariableType> Moead<DecisionVariableType>::GenerateNewSolution(int index) {
    Eigen::ArrayXi childrenIndex = selection->Select(crossover->GetParentNum(), neighborhoodIndexes.col(index));
    Eigen::ArrayXX<DecisionVariableType> parents = solutions.col(childrenIndex);
    Eigen::ArrayX<DecisionVariableType> newSolution = crossover->Cross(parents);
    mutation->Mutate(newSolution);
    return newSolution;
}

template <Number DecisionVariableType>
void Moead<DecisionVariableType>::UpdateIdealPoint(Eigen::ArrayXd objectiveSet) {
    idealPoint = idealPoint.min(objectiveSet);
}

template <Number DecisionVariableType>
void Moead<DecisionVariableType>::UpdateNeighboringSolutions(int index, Eigen::ArrayX<DecisionVariableType> solution,
                                                             Eigen::ArrayXd objectiveSet) {
    for (auto &&i : neighborhoodIndexes.col(index)) {
        double newSubObjective = decomposition->ComputeObjective(weightVectors.col(i), objectiveSet, idealPoint);
        double oldSubObjective = decomposition->ComputeObjective(weightVectors.col(i), objectiveSets.col(i), idealPoint);
        if (newSubObjective <= oldSubObjective) {
            solutions.col(i) = solution;
            objectiveSets.col(i) = objectiveSet;
        }
    }
}

}  // namespace Eacpp

// #include <chrono>
// #include <cmath>
// #include <eigen3/Eigen/Dense>
// #include <functional>
// #include <iostream>
// #include <memory>
// #include <numeric>
// #include <random>
// #include <tuple>
// #include <vector>

// template <typename T>
// std::vector<std::vector<T>> productRecurse(
//     const std::vector<T> &choices, const int repeat,
//     const std::vector<std::vector<T>> &result = std::vector<std::vector<T>>(1, std::vector<T>())) {
//     if (repeat == 0) {
//         return result;
//     }  // デカルト積が完成すれば答えを返す

//     std::vector<std::vector<T>> provisionalResult;  // 作成途中のデカルト積
//     std::cout << "start" << std::endl;
//     for (auto re : result) {
//         for (auto c : choices) {
//             std::vector<T> temp = re;
//             temp.push_back(c);
//             provisionalResult.push_back(temp);
//         }
//     }

//     return productRecurse(choices, repeat - 1, provisionalResult);
// }

// template <typename T>
// std::vector<std::vector<T>> product(const std::vector<T> &choices, const int repeat) {
//     return productRecurse(choices, repeat);
// }

// int main() {
//     int H = 9;
//     std::vector<int> v(H + 1);
//     std::iota(v.begin(), v.end(), 0);
//     auto res = product(v, 3);
//     res.erase(std::remove_if(res.begin(), res.end(), [&](std::vector<int> v) { return std::reduce(v.begin(), v.end()) != H;
//     }),
//               res.end());
//     for (auto r : res) {
//         for (auto i : r) {
//             std::cout << i << " ";
//         }
//         std::cout << std::endl;
//     }
//     std::cout << "size: " << res.size() << std::endl;

//     return 0;
// }