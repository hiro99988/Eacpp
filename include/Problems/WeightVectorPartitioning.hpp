#pragma once

#include <vector>

#include "Algorithms/MoeadInitializer.h"
#include "Algorithms/SimulatedAnnealing.hpp"
#include "Graph/SimpleGraph.hpp"
#include "Rng/Rng.h"
#include "Utils/Utils.h"

namespace Eacpp {
using SolutionType = std::vector<std::vector<std::vector<int>>>;

class PartitioningProblem : public SingleObjectiveProblem<SolutionType> {
   public:
    PartitioningProblem(int divisionsNumOfWeightVector, int objectivesNum,
                        int neighborhoodSize, int parallelSize)
        : _divisionsNumOfWeightVector(divisionsNumOfWeightVector),
          _objectivesNum(objectivesNum),
          _neighborhoodSize(neighborhoodSize),
          _parallelSize(parallelSize),
          _moeadInitializer() {}

    /// @brief 目的関数を計算する
    /// @param solution 解
    /// @return 解の目的値
    /// @details 目的関数は、グラフの平均次数を計算する
    double ComputeObjective(const SolutionType& solution) override;

    /// @brief 解に基づいてグラフを生成する
    /// @param solution 解
    /// @return 生成されたグラフ
    SimpleGraph GenerateGraph(const SolutionType& solution);

   private:
    int _divisionsNumOfWeightVector;
    int _objectivesNum;
    int _neighborhoodSize;
    int _parallelSize;
    MoeadInitializer _moeadInitializer;

    /// @brief MoeadInitializerを利用して並列MOEA/Dの隣接リストを生成する
    void GenerateAdjacencyList(const SolutionType& solution,
                               std::vector<int>& outRanksToSendAtInitialization,
                               std::vector<int>& outRanksToSendCounts,
                               std::vector<int>& outNeighboringRanks,
                               std::vector<int>& outNeighboringRankCounts);
};

class NeighborGen : public NeighborGenerator<SolutionType> {
   public:
    NeighborGen() : _rng() {}

    /// @brief 近傍解を生成する
    /// @param currentSolution 現在の解
    /// @return 新しい近傍解
    SolutionType Generate(const SolutionType& currentSolution) override;

   private:
    Rng _rng;

    /// @brief 分割 node の距離行列を計算する
    /// @param solution 解
    /// @param node 分割インデックス
    /// @return 距離行列
    /// @details
    /// 任意の分割に含まれる全ての重みベクトルと任意の分割に含まれる全ての
    /// 重みベクトルの平均距離を計算し，分割同士の距離行列を生成する．
    /// 具体的には，分割同士の距離行列の要素は，分割iに含まれる重みベクトル
    /// と分割jに含まれる重みベクトルの平均距離を計算する．
    /// そのため，分割iに含まれる重みベクトルの数をn_i，分割jに含まれる
    /// 重みベクトルの数をn_jとすると，分割同士の距離行列の要素[i][j]は
    /// (1 / (n_i * n_j)) * Σ_{k=1}^{n_i} Σ_{l=1}^{n_j} dist(wv_i[k], wv_j[l])
    /// となる．
    std::vector<double> ComputeDistanceMatrix(const SolutionType& solution,
                                              std::size_t node);

    /// @brief 分割インデックスを重み付きランダムに選択する
    /// @param avgDist 距離行列
    /// @return 選択された分割インデックス
    /// @details
    /// 距離が小さいほど選ばれやすくするため，重みを逆数で計算する．
    /// 具体的には，距離行列の要素[i][j]が小さいほど，分割iが選ばれやすくなる．
    std::size_t SelectIndexByWeightedRandom(const std::vector<double>& avgDist);
};

}  // namespace Eacpp