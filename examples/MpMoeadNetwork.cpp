#include <iostream>
#include <vector>

#include "Algorithms/MoeadInitializer.h"
#include "Individual.h"

using namespace Eacpp;

int main(int argc, char* argv[]) {
    int divisionsNumOfWeightVector = 199;
    int objectivesNum = 2;
    int neighborhoodSize = 21;
    int parallelSize = 50;

    if (argc == 5) {
        divisionsNumOfWeightVector = std::stoi(argv[1]);
        objectivesNum = std::stoi(argv[2]);
        neighborhoodSize = std::stoi(argv[3]);
        parallelSize = std::stoi(argv[4]);
    }

    MoeadInitializer moeadInitializer;
    std::vector<int> outInternalIndividualIndexes;
    std::vector<double> outInternalWeightVectors;
    std::vector<int> outInternalNeighborhoods;
    std::vector<int> outInternalIndividualCounts;
    std::vector<int> outExternalIndividualIndexes;
    std::vector<int> outExternalIndividualRanks;
    std::vector<double> outExternalWeightVectors;
    std::vector<int> outExternalIndividualCounts;
    std::vector<int> outRanksToSendAtInitialization;
    std::vector<int> outRanksToSendCounts;
    std::vector<int> outNeighboringRanks;
    std::vector<int> outNeighboringRankCounts;
    moeadInitializer.InitializeParallelMoead(
        divisionsNumOfWeightVector, objectivesNum, neighborhoodSize,
        parallelSize, outInternalIndividualIndexes, outInternalWeightVectors,
        outInternalNeighborhoods, outInternalIndividualCounts,
        outExternalIndividualIndexes, outExternalIndividualRanks,
        outExternalWeightVectors, outExternalIndividualCounts,
        outRanksToSendAtInitialization, outRanksToSendCounts,
        outNeighboringRanks, outNeighboringRankCounts);

    // 生成した内部個体数と母集団サイズを比較
    auto populationSize = moeadInitializer.CalculatePopulationSize(
        divisionsNumOfWeightVector, objectivesNum);
    std::cout << "internal individuals: " << "size "
              << outInternalIndividualIndexes.size();
    if (outInternalIndividualIndexes.size() != populationSize) {
        std::cout << " != ";
    } else {
        std::cout << " == ";
    }
    std::cout << "population size " << populationSize << std::endl;
    // 内部個体情報を表示
    for (int i = 0, count = 0; i < outInternalIndividualCounts.size();
         count += outInternalIndividualCounts[i], ++i) {
        // ランクと割り当てられた個体数を表示
        int size = outInternalIndividualCounts[i];
        std::cout << "rank: " << i << " size: " << size << std::endl;
        // 内部個体のインデックスを表示
        std::cout << "index: ";
        for (int j = 0; j < size; ++j) {
            std::cout << outInternalIndividualIndexes[count + j] << " ";
        }
        std::cout << std::endl;
        // 内部個体の重みベクトルを表示
        std::cout << "weight: ";
        for (int j = count * objectivesNum; j < (count + size) * objectivesNum;
             j += objectivesNum) {
            std::cout << "[";
            for (int k = 0; k < objectivesNum; ++k) {
                std::cout << outInternalWeightVectors[j + k];
                if (k != objectivesNum - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << "], ";
        }
        std::cout << std::endl;
    }
    // 生成した内部個体の近傍の合計数と真の近傍の合計数を比較
    std::cout << "internalIndividualNeighborhoods: size "
              << outInternalNeighborhoods.size();
    if (outInternalNeighborhoods.size() !=
        (populationSize * neighborhoodSize)) {
        std::cout << " != ";
    } else {
        std::cout << " == ";
    }
    std::cout << "true size " << (populationSize * neighborhoodSize)
              << std::endl;
    // 内部個体の近傍情報を表示
    for (int i = 0, count = 0; i < outExternalIndividualCounts.size();
         count += outExternalIndividualCounts[i], ++i) {
        // ランクと割り当てられた近傍数を表示
        int size = outExternalIndividualCounts[i];
        std::cout << "rank: " << i << " size: " << size << std::endl;
        // 内部個体の近傍インデックスを表示
        std::cout << "index: ";
        for (int j = 0; j < size; ++j) {
            std::cout << outExternalIndividualIndexes[count + j] << " ";
        }
        std::cout << std::endl;
        // 内部個体の近傍が所属するランクを表示
        std::cout << " rank: ";
        for (int j = 0; j < size; ++j) {
            std::cout << outExternalIndividualRanks[count + j] << " ";
        }
        std::cout << std::endl;
        // 内部個体の近傍の重みベクトルを表示
        std::cout << "weight: ";
        for (int j = count * objectivesNum; j < (count + size) * objectivesNum;
             j += objectivesNum) {
            std::cout << "[";
            for (int k = 0; k < objectivesNum; ++k) {
                std::cout << outInternalWeightVectors[j + k];
                if (k != objectivesNum - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << "], ";
        }
        std::cout << std::endl;
    }

    std::cout << "ranksToSend: " << std::endl;
    // ranksToSendの平均サイズ
    int sum = 0;
    int min = outRanksToSendCounts[0];
    int max = outRanksToSendCounts[0];
    for (auto&& i : outRanksToSendCounts) {
        sum += i;
        if (i < min) {
            min = i;
        }
        if (i > max) {
            max = i;
        }
    }
    std::cout << "ranksToSend min size: " << min << std::endl;
    std::cout << "ranksToSend max size: " << max << std::endl;
    std::cout << "ranksToSend average size: "
              << (double)sum / outRanksToSendCounts.size() << std::endl;
    for (int i = 0, count = 0; i < outRanksToSendCounts.size();
         count += outRanksToSendCounts[i], ++i) {
        std::cout << "[";
        for (int j = 0; j < outRanksToSendCounts[i]; ++j) {
            std::cout << outRanksToSendAtInitialization[count + j];
            if (j != outRanksToSendCounts[i] - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]," << std::endl;
    }

    std::cout << "neighboringRanks:" << std::endl;
    // neighboringRanksの平均サイズ
    sum = 0;
    min = outNeighboringRankCounts[0];
    max = outNeighboringRankCounts[0];
    for (auto&& i : outNeighboringRankCounts) {
        sum += i;
        if (i < min) {
            min = i;
        }
        if (i > max) {
            max = i;
        }
    }
    std::cout << "neighboringRanks min size: " << min << std::endl;
    std::cout << "neighboringRanks max size: " << max << std::endl;
    std::cout << "neighboringRanks average size: "
              << (double)sum / outNeighboringRankCounts.size() << std::endl;
    for (int i = 0, count = 0; i < outNeighboringRankCounts.size();
         count += outNeighboringRankCounts[i], ++i) {
        int size = outNeighboringRankCounts[i];
        std::cout << "[";
        for (int j = 0; j < size; ++j) {
            std::cout << outNeighboringRanks[count + j];
            if (j != size - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]," << std::endl;
    }

    return 0;
}