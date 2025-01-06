#pragma once

#include <algorithm>
#include <eigen3/Eigen/Core>
#include <numeric>
#include <ranges>
#include <vector>

#include "Utils/EigenUtils.h"
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

   private:
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
};

}  // namespace Eacpp