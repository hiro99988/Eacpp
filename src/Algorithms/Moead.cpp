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

// Eigen::ArrayXXd GenerateWeightVectors(int H, int objectiveNum) {
//     Eigen::ArrayXXd takedSetTop = Eigen::ArrayXd::LinSpaced(H + 1, 0, H);
//     takedSetTop.resize(1, H + 1);
//     return takedSetTop;
//     // Eigen::ArrayXd pair = H - takedSetTop;
//     // Eigen::ArrayXXd product(2, H + 1);
//     // product << takedSetTop, pair;
//     // return product;
//     // for (int i = 0; i < objectiveNum; i++) {
//     //     product.col(i) = takedSetTop;
//     // }
//     // Eigen::ArrayXXd wvTop = product.rowwise().sum();
//     // return wvTop / double(H);
// }

// // def generate_weight_vector(m, H):
// //     taked_set_top = [i for i in range(H + 1)]
// //     product = np.array(list(itertools.product(taked_set_top, repeat=m)))
// //     wv_top = product[np.sum(product, axis=1) == H]
// //     wv = wv_top / H
// //     return wv

// /// @brief 組み合わせC(n, k)を計算する
// /// @tparam Iterator
// /// @param first
// /// @param k
// /// @param last
// /// @return
// template <typename Iterator>
// bool next_combination(const Iterator first, Iterator k, const Iterator last) {
//     if ((first == last) || (first == k) || (last == k)) return false;
//     Iterator itr1 = first;
//     Iterator itr2 = last;
//     ++itr1;
//     if (last == itr1) return false;
//     itr1 = last;
//     --itr1;
//     itr1 = k;
//     --itr2;
//     while (first != itr1) {
//         if (*--itr1 < *itr2) {
//             Iterator j = k;
//             while (!(*itr1 < *j)) ++j;
//             std::iter_swap(itr1, j);
//             ++itr1;
//             ++j;
//             itr2 = k;
//             std::rotate(itr1, j, last);
//             while (last != j) {
//                 ++j;
//                 ++itr2;
//             }
//             std::rotate(k, itr2, last);
//             return true;
//         }
//     }
//     std::rotate(first, k, last);
//     return false;
// }

// /// @brief 重複なし順列
// /// @tparam T
// /// @param v
// /// @param k
// /// @return
// template <typename T>
// std::vector<std::vector<T>> Permutation(std::vector<T> &v, int k) {
//     std::vector<std::vector<T>> result;
//     std::vector<T> tmp(k);
//     do {
//         std::copy(v.begin(), v.begin() + k, tmp.begin());
//         do {
//             result.push_back(tmp);
//         } while (std::prev_permutation(tmp.begin(), tmp.end()));
//     } while (next_combination(v.begin(), v.begin() + k, v.end()));
//     return result;
// }

// /// @brief 重複ありの順列
// /// @tparam container_type
// /// @param choices
// /// @param n
// /// @return
// template <typename container_type>
// std::vector<std::vector<typename container_type::value_type>> combWithReplace(container_type const &choices, size_t n) {
//     using value_type = typename container_type::value_type;
//     using selected_t = std::vector<value_type>;
//     using itor_t = typename container_type::const_iterator;
//     struct impl {                                  // lambda で再帰は面倒なので クラスにする
//         std::vector<std::vector<value_type>> &r_;  // コピーを避けるために参照を持つ
//         impl(std::vector<std::vector<value_type>> &r) : r_(r) {}
//         void append(selected_t &s, itor_t b, itor_t e, size_t n) {
//             if (n == 0) {
//                 r_.push_back(s);
//             } else {
//                 for (auto it = b; it != e; ++it) {
//                     s.push_back(*it);
//                     append(s, it, e, n - 1);
//                     s.pop_back();
//                 }
//             }
//         };
//     };
//     std::vector<std::vector<value_type>> r;
//     impl o{r};
//     selected_t e;
//     e.reserve(n);
//     o.append(e, std::cbegin(choices), std::cend(choices), n);
//     return r;
// }

// template <typename T>
// std::vector<std::vector<T>> productRecurse(const std::vector<T> &choices, int repeat, std::vector<std::vector<T>> result) {
//     if (repeat == 0) {
//         return result;
//     }  // デカルト積が完成すれば答えを返す

//     std::vector<std::vector<T>> provisionalResult;  // 作成途中のデカルト積

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
// std::vector<std::vector<T>> product(const std::vector<T> &choices, int repeat) {
//     std::vector<std::vector<T>> emptyResult(1, std::vector<T>());  // 組み合わせを格納するための空のリスト

//     return productRecurse(choices, repeat, emptyResult);
// }

// int main() {
//     int H = 5;
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