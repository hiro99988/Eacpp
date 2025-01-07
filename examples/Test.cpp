#include <iostream>
#include <vector>

#include "Algorithms/MoeadInitializer.h"
#include "Individual.h"

using namespace Eacpp;

int main() {
    MoeadInitializer moeadInitializer;
    int divisionsNumOfWeightVector = 199;
    int objectivesNum = 2;
    int neighborhoodSize = 21;
    int parallelSize = 50;
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

    std::cout << "internal" << std::endl;
    for (int i = 0, count = 0; i < outInternalIndividualCounts.size();
         count += outInternalIndividualCounts[i], ++i) {
        int size = outInternalIndividualCounts[i];
        std::cout << "rank: " << i << " size: " << size << std::endl;
        for (int j = 0; j < size; ++j) {
            std::cout << outInternalIndividualIndexes[count + j] << " ";
        }
        std::cout << std::endl;
        for (int j = count * objectivesNum; j < (count + size) * objectivesNum;
             j += objectivesNum) {
            for (int k = 0; k < objectivesNum; ++k) {
                std::cout << outInternalWeightVectors[j + k] << " ";
            }
        }
        std::cout << std::endl;
    }
    std::cout << "internalIndividualNeighborhoods size: "
              << outInternalNeighborhoods.size() << " == "
              << (outInternalIndividualIndexes.size() * neighborhoodSize) << " "
              << std::boolalpha
              << (outInternalNeighborhoods.size() ==
                  (outInternalIndividualIndexes.size() * neighborhoodSize))
              << std::endl;

    std::cout << "external" << std::endl;
    for (int i = 0, count = 0; i < outExternalIndividualCounts.size();
         count += outExternalIndividualCounts[i], ++i) {
        int size = outExternalIndividualCounts[i];
        std::cout << "rank: " << i << " size: " << size << std::endl;
        for (int j = 0; j < size; ++j) {
            std::cout << outExternalIndividualIndexes[count + j] << " ";
        }
        std::cout << std::endl;
        for (int j = 0; j < size; ++j) {
            std::cout << outExternalIndividualRanks[count + j] << " ";
        }
        std::cout << std::endl;
        for (int j = count * objectivesNum; j < (count + size) * objectivesNum;
             j += objectivesNum) {
            for (int k = 0; k < objectivesNum; ++k) {
                std::cout << outExternalWeightVectors[j + k] << " ";
            }
        }
        std::cout << std::endl;
    }

    std::cout << "ranksToSend" << std::endl;
    for (int i = 0, count = 0; i < outRanksToSendCounts.size();
         count += outRanksToSendCounts[i], ++i) {
        int size = outRanksToSendCounts[i];
        std::cout << "rank: " << i << " size: " << size << std::endl;
        for (int j = 0; j < size; ++j) {
            std::cout << outRanksToSendAtInitialization[count + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "neighboringRanks" << std::endl;
    for (int i = 0, count = 0; i < outNeighboringRankCounts.size();
         count += outNeighboringRankCounts[i], ++i) {
        int size = outNeighboringRankCounts[i];
        std::cout << "rank: " << i << " size: " << size << std::endl;
        for (int j = 0; j < size; ++j) {
            std::cout << outNeighboringRanks[count + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}