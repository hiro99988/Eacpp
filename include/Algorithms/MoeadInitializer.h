#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <deque>
#include <numeric>
#include <ranges>
#include <set>
#include <vector>

#include "Utils/EigenUtils.h"
#include "Utils/MpiUtils.h"
#include "Utils/Utils.h"

namespace Eacpp {

class MoeadInitializer {
   public:
    int CalculatePopulationSize(int divisionsNumOfWeightVector,
                                int objectivesNum) const {
        int n = divisionsNumOfWeightVector + objectivesNum - 1;
        int r = objectivesNum - 1;
        return Combination(n, r);
    }

    std::vector<std::vector<double>> GenerateWeightVectors(
        int divisionsNumOfWeightVector, int objectivesNum) const {
        // 0からdivisionsNumOfWeightVectorまでの数列のobjectivesNum-順列を生成する
        std::vector<double> numerators(divisionsNumOfWeightVector + 1);
        std::iota(numerators.begin(), numerators.end(), 0);
        std::vector<std::vector<double>> permutations =
            Product(numerators, objectivesNum);

        // 各組み合わせの合計がdivisionsNumOfWeightVectorになるものを抽出する
        std::vector<std::vector<double>> validCombinations;
        validCombinations.reserve(permutations.size());
        for (const auto& v : permutations) {
            if (std::reduce(v.begin(), v.end()) == divisionsNumOfWeightVector) {
                validCombinations.push_back(v);
            }
        }

        // 各値をdivisionsNumOfWeightVectorで割り，重みベクトルを生成する
        for (auto& combination : validCombinations) {
            for (auto& i : combination) {
                i /= divisionsNumOfWeightVector;
            }
        }

        return validCombinations;
    }

    std::vector<Eigen::ArrayXd> GenerateWeightVectorsEigenArray(
        int divisionsNumOfWeightVector, int objectivesNum) const {
        auto weightVectors =
            GenerateWeightVectors(divisionsNumOfWeightVector, objectivesNum);

        std::vector<Eigen::ArrayXd> weightVectorsEigenArray;
        weightVectorsEigenArray.reserve(weightVectors.size());
        for (auto&& v : weightVectors) {
            Eigen::ArrayXd weightVector =
                Eigen::Map<Eigen::ArrayXd>(v.data(), v.size());
            weightVectorsEigenArray.push_back(std::move(weightVector));
        }

        return weightVectorsEigenArray;
    }

    std::vector<double> GenerateWeightVectors1d(int divisionsNumOfWeightVector,
                                                int objectivesNum) const {
        auto weightVectors =
            GenerateWeightVectors(divisionsNumOfWeightVector, objectivesNum);

        std::vector<double> weightVectors1d;
        weightVectors1d.reserve(weightVectors.size() * objectivesNum);
        for (auto&& v : weightVectors) {
            weightVectors1d.insert(weightVectors1d.end(), v.begin(), v.end());
        }

        return weightVectors1d;
    }

    std::vector<std::vector<int>> CalculateNeighborhoods2d(
        int neighborhoodSize,
        const std::vector<std::vector<double>>& weightVectors) const {
        auto euclideanDistanceMatrix =
            CalculateEuclideanDistanceMatrix(weightVectors);

        std::vector<std::vector<int>> neighborhoods;
        neighborhoods.reserve(weightVectors.size());
        for (auto&& euclideanDistances : euclideanDistanceMatrix) {
            neighborhoods.push_back(
                CalculateNeighborhood(euclideanDistances, neighborhoodSize));
        }

        return neighborhoods;
    }

    std::vector<std::vector<int>> CalculateNeighborhoods2d(
        int neighborhoodSize,
        const std::vector<Eigen::ArrayXd>& weightVectors) const {
        auto euclideanDistanceMatrix =
            CalculateEuclideanDistanceMatrix(weightVectors);

        std::vector<std::vector<int>> neighborhoods;
        neighborhoods.reserve(weightVectors.size());
        for (auto&& euclideanDistances : euclideanDistanceMatrix) {
            neighborhoods.push_back(
                CalculateNeighborhood(euclideanDistances, neighborhoodSize));
        }

        return neighborhoods;
    }

    std::vector<int> CalculateNeighborhoods1d(
        int objectivesNum, int neighborhoodSize,
        const std::vector<double>& weightVectors) const {
        auto euclideanDistanceMatrix =
            CalculateEuclideanDistanceMatrix(objectivesNum, weightVectors);

        std::vector<int> neighborhoods;
        neighborhoods.reserve(weightVectors.size() * neighborhoodSize);
        for (auto&& euclideanDistances : euclideanDistanceMatrix) {
            auto neighborhood =
                CalculateNeighborhood(euclideanDistances, neighborhoodSize);
            neighborhoods.insert(neighborhoods.end(),
                                 std::make_move_iterator(neighborhood.begin()),
                                 std::make_move_iterator(neighborhood.end()));
        }

        return neighborhoods;
    }

    void GenerateWeightVectorsAndNeighborhoods(
        int divisionsNumOfWeightVector, int objectivesNum, int neighborhoodSize,
        std::vector<Eigen::ArrayXd>& outWeightVectors,
        std::vector<std::vector<int>>& outNeighborhoods) const {
        outWeightVectors = GenerateWeightVectorsEigenArray(
            divisionsNumOfWeightVector, objectivesNum);
        outNeighborhoods =
            CalculateNeighborhoods2d(neighborhoodSize, outWeightVectors);
    }

    void GenerateWeightVectorsAndNeighborhoods(
        int divisionsNumOfWeightVector, int objectivesNum, int neighborhoodSize,
        std::vector<double>& outWeightVectors,
        std::vector<int>& outNeighborhoods) const {
        outWeightVectors =
            GenerateWeightVectors1d(divisionsNumOfWeightVector, objectivesNum);
        outNeighborhoods = CalculateNeighborhoods1d(
            objectivesNum, neighborhoodSize, outWeightVectors);
    }

    void GenerateWeightVectorsAndNeighborhoods(
        int divisionsNumOfWeightVector, int objectivesNum, int neighborhoodSize,
        std::vector<std::vector<double>>& outWeightVectors,
        std::vector<std::vector<int>>& outNeighborhoods) const {
        outWeightVectors =
            GenerateWeightVectors(divisionsNumOfWeightVector, objectivesNum);
        outNeighborhoods =
            CalculateNeighborhoods2d(neighborhoodSize, outWeightVectors);
    }

    /* TODO: 重みベクトル internalの近傍インデックス internalのインデックス
     * externalのインデックス externalの重みベクトル 近傍のランク */

    void InitializeParallelMoead(
        int divisionsNumOfWeightVector, int objectivesNum, int neighborhoodSize,
        int parallelSize, std::vector<int>& outInternalIndividualIndexes,
        std::vector<double>& outInternalWeightVectors,
        std::vector<int>& outInternalNeighborhoods,
        std::vector<int>& outInternalIndividualCounts,
        std::vector<int>& outExternalIndividualIndexes,
        std::vector<int>& outExternalIndividualRanks,
        std::vector<double>& outExternalWeightVectors,
        std::vector<int>& outExternalIndividualCounts,
        std::vector<int>& outRanksToSendAtInitialization,
        std::vector<int>& outRanksToSendCounts,
        std::vector<int>& outNeighboringRanks,
        std::vector<int>& outNeighboringRankCounts) {
        int totalPopulationSize =
            CalculatePopulationSize(divisionsNumOfWeightVector, objectivesNum);

        auto vectors = GenerateWeightVectorDivisions(divisionsNumOfWeightVector,
                                                     objectivesNum);
        auto partitions = NearestNeighborPartitioning(
            parallelSize, divisionsNumOfWeightVector, vectors);
        // auto partitions = LinearPartitioning(
        //     parallelSize, divisionsNumOfWeightVector, vectors);

        std::vector<std::vector<int>> internalIndividualIndexes;
        internalIndividualIndexes.reserve(parallelSize);
        for (int i = 0, count = 0; i < partitions.size();
             count += partitions[i].size(), ++i) {
            internalIndividualIndexes.emplace_back();
            internalIndividualIndexes[i].reserve(partitions[i].size());
            for (int j = count; j < count + partitions[i].size(); ++j) {
                internalIndividualIndexes[i].push_back(j);
            }
        }

        auto weightVectors = GenerateWeightVectors(
            partitions, divisionsNumOfWeightVector, totalPopulationSize);
        auto neighborhoods =
            CalculateNeighborhoods2d(neighborhoodSize, weightVectors);

        // std::vector<std::vector<double>> weightVectors;
        // std::vector<std::vector<int>> neighborhoods;
        // GenerateWeightVectorsAndNeighborhoods(divisionsNumOfWeightVector,
        //                                       objectivesNum,
        //                                       neighborhoodSize,
        //                                       weightVectors, neighborhoods);

        // // 各ランクのインデックスを生成する
        // auto internalIndividualIndexes =
        //     GenerateAllNodeIndexes(totalPopulationSize, parallelSize);

        // internalIndividualIndexesを1次元に変換，個数をカウント
        outInternalIndividualIndexes.reserve(
            internalIndividualIndexes.size() *
            internalIndividualIndexes[0].size());
        outInternalIndividualCounts.reserve(internalIndividualIndexes.size());
        for (const auto& indexes : internalIndividualIndexes) {
            outInternalIndividualIndexes.insert(
                outInternalIndividualIndexes.end(), indexes.begin(),
                indexes.end());
            outInternalIndividualCounts.push_back(indexes.size());
        }

        // 内部の重みベクトルと近傍のインデックスをまとめる
        outInternalWeightVectors.reserve(outInternalIndividualIndexes.size() *
                                         objectivesNum);
        outInternalNeighborhoods.reserve(outInternalIndividualIndexes.size() *
                                         neighborhoodSize);
        for (auto&& i : outInternalIndividualIndexes) {
            outInternalWeightVectors.insert(outInternalWeightVectors.end(),
                                            weightVectors[i].begin(),
                                            weightVectors[i].end());
            outInternalNeighborhoods.insert(outInternalNeighborhoods.end(),
                                            neighborhoods[i].begin(),
                                            neighborhoods[i].end());
        }

        // 個体が所属するランクを計算する
        std::vector<int> individualRanks(totalPopulationSize);
        for (int i = 0; i < internalIndividualIndexes.size(); ++i) {
            for (auto&& j : internalIndividualIndexes[i]) {
                individualRanks[j] = i;
            }
        }

        // 各ランクの外部のインデックス，初期化時に送信するランクを計算する
        std::vector<std::vector<int>> externalRanks;
        std::size_t externalRankSize = 0;
        externalRanks.reserve(parallelSize);
        outExternalIndividualCounts.reserve(parallelSize);
        for (int i = 0; i < parallelSize; ++i) {
            // ランクiの全ての個体の近傍のインデックスをまとめる
            std::vector<int> externalIndexes;
            externalIndexes.reserve(internalIndividualIndexes[i].size() *
                                    neighborhoodSize);
            for (auto&& j : internalIndividualIndexes[i]) {
                externalIndexes.insert(externalIndexes.end(),
                                       neighborhoods[j].begin(),
                                       neighborhoods[j].end());
            }

            RemoveDuplicates(externalIndexes);

            // 内部のインデックスを削除
            std::erase_if(externalIndexes, [&](int index) {
                return std::ranges::find(internalIndividualIndexes[i], index) !=
                       internalIndividualIndexes[i].end();
            });

            // ランクiの外部のインデックスをまとめる，個数をカウント
            outExternalIndividualIndexes.insert(
                outExternalIndividualIndexes.end(), externalIndexes.begin(),
                externalIndexes.end());
            outExternalIndividualCounts.push_back(externalIndexes.size());

            // 外部個体が所属するランク，ランクiの近傍のランクを計算する
            std::vector<int> ranks;
            ranks.reserve(externalIndexes.size());
            for (auto&& j : externalIndexes) {
                outExternalIndividualRanks.push_back(individualRanks[j]);
                ranks.push_back(individualRanks[j]);
            }
            RemoveDuplicates(ranks);

            // 初期化時に送信するランク，近傍のランクの計算に使うために保存する
            externalRankSize += ranks.size();
            externalRanks.push_back(std::move(ranks));
        }

        // 外部の重みベクトルをまとめる
        outExternalWeightVectors.reserve(outExternalIndividualIndexes.size() *
                                         objectivesNum);
        for (auto&& i : outExternalIndividualIndexes) {
            outExternalWeightVectors.insert(outExternalWeightVectors.end(),
                                            weightVectors[i].begin(),
                                            weightVectors[i].end());
        }

        // 初期化時に送信するランクを計算する
        std::vector<std::vector<int>> ranksToSend(parallelSize,
                                                  std::vector<int>());
        for (int i = 0; i < parallelSize; ++i) {
            for (auto&& j : externalRanks[i]) {
                ranksToSend[j].push_back(i);
            }
        }

        // 1次元に変換，個数をカウント
        outRanksToSendAtInitialization.reserve(externalRankSize);
        outRanksToSendCounts.reserve(parallelSize);
        for (std::size_t i = 0; i < ranksToSend.size(); ++i) {
            outRanksToSendAtInitialization.insert(
                outRanksToSendAtInitialization.end(), ranksToSend[i].begin(),
                ranksToSend[i].end());
            outRanksToSendCounts.push_back(ranksToSend[i].size());
        }

        // 近傍のランクを計算する
        outNeighboringRankCounts = std::vector<int>(parallelSize);
        for (int i = 0; i < parallelSize; ++i) {
            // for (auto&& j : externalRanks[i]) {
            // if (std::ranges::find(externalRanks[j], i) !=
            //     externalRanks[j].end()) {
            //     outNeighboringRanks.push_back(j);
            //     outNeighboringRankCounts[i]++;
            // }
            // }
            outNeighboringRanks.insert(outNeighboringRanks.end(),
                                       externalRanks[i].begin(),
                                       externalRanks[i].end());
            outNeighboringRankCounts[i] = externalRanks[i].size();
        }
    }

   private:
    std::vector<std::vector<int>> GenerateWeightVectorDivisions(
        int divisionsNumOfWeightVector, int objectivesNum) const {
        // 0からdivisionsNumOfWeightVectorまでの数列のobjectivesNum-順列を生成する
        std::vector<int> numerators(divisionsNumOfWeightVector + 1);
        std::iota(numerators.begin(), numerators.end(), 0);
        auto permutations = Product(numerators, objectivesNum);

        // 各組み合わせの合計がdivisionsNumOfWeightVectorになるものを抽出する
        std::vector<std::vector<int>> validCombinations;
        validCombinations.reserve(permutations.size());
        for (const auto& v : permutations) {
            if (std::reduce(v.begin(), v.end()) == divisionsNumOfWeightVector) {
                validCombinations.push_back(v);
            }
        }

        return validCombinations;
    }

    std::vector<std::vector<double>> CalculateEuclideanDistanceMatrix(
        const std::vector<Eigen::ArrayXd>& weightVectors) const {
        int size = weightVectors.size();
        std::vector<std::vector<double>> euclideanDistances(
            size, std::vector<double>(size));
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                euclideanDistances[i][j] = CalculateSquaredEuclideanDistance(
                    weightVectors[i], weightVectors[j]);
            }
        }

        return euclideanDistances;
    }

    std::vector<std::vector<double>> CalculateEuclideanDistanceMatrix(
        int objectivesNum, const std::vector<double>& weightVectors) const {
        int size = weightVectors.size() / objectivesNum;
        std::vector<std::vector<double>> euclideanDistances(
            size, std::vector<double>(size));
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                double distance = 0.0;
                for (int k = 0; k < objectivesNum; k++) {
                    distance +=
                        std::pow(weightVectors[i * objectivesNum + k] -
                                     weightVectors[j * objectivesNum + k],
                                 2);
                }

                euclideanDistances[i][j] = distance;
            }
        }

        return euclideanDistances;
    }

    std::vector<std::vector<double>> CalculateEuclideanDistanceMatrix(
        const std::vector<std::vector<double>>& weightVectors) const {
        int size = weightVectors.size();
        std::vector<std::vector<double>> euclideanDistances(
            size, std::vector<double>(size));
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                euclideanDistances[i][j] = CalculateSquaredEuclideanDistance(
                    weightVectors[i], weightVectors[j]);
            }
        }

        return euclideanDistances;
    }

    std::vector<int> CalculateNeighborhood(
        std::vector<double>& euclideanDistances, int neighborhoodSize) const {
        auto sortedIndexes = ArgSort(euclideanDistances);
        return std::vector<int>(sortedIndexes.begin(),
                                sortedIndexes.begin() + neighborhoodSize);
    }

    std::vector<int> LexicographicalMin(
        const std::vector<std::vector<int>>& vectors,
        const std::set<std::vector<int>>& visitedVectors) {
        const std::vector<int>* min = nullptr;
        for (const auto& v : vectors) {
            if (visitedVectors.find(v) == visitedVectors.end()) {
                if (min == nullptr) {
                    min = &v;
                } else if (v < *min) {
                    min = &v;
                }
            }
        }

        if (min == nullptr) {
            return {};
        } else {
            return *min;
        }
    }

    std::vector<std::vector<int>> nearest(const std::vector<int>& vector,
                                          int divisionsNumOfWeightVector) {
        std::vector<std::vector<int>> neighbors;
        for (std::size_t i = 0; i < vector.size(); ++i) {
            for (std::size_t j = 0; j < vector.size(); ++j) {
                if (i != j && vector[i] > 0 &&
                    vector[j] < divisionsNumOfWeightVector) {
                    std::vector<int> neighbor = vector;
                    neighbor[i] -= 1;
                    neighbor[j] += 1;
                    neighbors.push_back(std::move(neighbor));
                }
            }
        }

        std::sort(neighbors.begin(), neighbors.end());

        return neighbors;
    }

    std::vector<std::vector<std::vector<int>>> NearestNeighborPartitioning(
        int partitionsNum, int divisionsNumOfWeightVector,
        std::vector<std::vector<int>>& vectors) {
        std::set<std::vector<int>> visitedVectors;
        std::vector<std::vector<std::vector<int>>> partitions;
        partitions.reserve(partitionsNum);
        std::deque<std::vector<int>> queue;
        std::size_t vectorsSize = vectors.size();
        int l = 0;

        for (int i = 0; i < partitionsNum; ++i) {
            int partitionSize =
                CalculateNodeWorkload(vectorsSize, i, partitionsNum);
            std::vector<std::vector<int>> currentPartition;
            currentPartition.reserve(partitionSize);
            while (currentPartition.size() != partitionSize) {
                std::vector<int> root;
                if (l == 0 || queue.empty()) {
                    root = LexicographicalMin(vectors, visitedVectors);
                    if (root.empty()) {
                        break;
                    }
                    l = root.back();
                } else {
                    root = *std::min_element(queue.begin(), queue.end());
                }

                queue = {root};
                while (currentPartition.size() < partitionSize) {
                    if (queue.empty()) {
                        break;
                    }

                    auto current = queue.front();
                    queue.pop_front();

                    if (visitedVectors.find(current) != visitedVectors.end()) {
                        continue;
                    }

                    currentPartition.push_back(current);
                    visitedVectors.insert(current);
                    l = std::min(l, current.back());

                    for (const auto& neighbor :
                         nearest(current, divisionsNumOfWeightVector)) {
                        if (visitedVectors.find(neighbor) ==
                                visitedVectors.end() &&
                            std::find(queue.begin(), queue.end(), neighbor) ==
                                queue.end()) {
                            queue.push_back(neighbor);
                        }
                    }
                }
            }

            partitions.push_back(std::move(currentPartition));
        }

        return partitions;
    }

    std::vector<std::vector<std::vector<int>>> LinearPartitioning(
        int partitionsNum, int divisionsNumOfWeightVector,
        const std::vector<std::vector<int>>& vectors) {
        auto sortedVectors = vectors;
        std::sort(sortedVectors.begin(), sortedVectors.end());
        std::vector<std::vector<std::vector<int>>> partitions;
        partitions.reserve(partitionsNum);
        auto partitionSizes =
            CalculateNodeWorkloads(sortedVectors.size(), partitionsNum);
        for (int i = 0, count = 0; i < partitionsNum;
             count += partitionSizes[i], ++i) {
            std::vector<std::vector<int>> partition;
            partition.reserve(partitionSizes[i]);
            for (int j = count; j < count + partitionSizes[i]; ++j) {
                partition.push_back(sortedVectors[j]);
            }
            partitions.push_back(std::move(partition));
        }

        return partitions;
    }

    std::vector<std::vector<double>> GenerateWeightVectors(
        std::vector<std::vector<std::vector<int>>> partitioins,
        int divisionsNumOfWeightVector, std::size_t size) {
        std::vector<std::vector<double>> weightVectors;
        weightVectors.reserve(size);
        for (const auto& partition : partitioins) {
            for (const auto& vector : partition) {
                weightVectors.emplace_back(vector.begin(), vector.end());
            }
        }

        for (auto& vector : weightVectors) {
            for (auto& i : vector) {
                i /= divisionsNumOfWeightVector;
            }
        }

        return weightVectors;
    }
};

}  // namespace Eacpp