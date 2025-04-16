#pragma once

#include <fstream>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "Algorithms/MpMoead.h"
#include "Crossovers/ICrossover.h"
#include "Decompositions/IDecomposition.h"
#include "Individual.h"
#include "Mutations/IMutation.h"
#include "Problems/IProblem.h"
#include "Repairs/IRepair.h"
#include "Samplings/ISampling.h"
#include "Selections/ISelection.h"
#include "Utils/FileUtils.h"

namespace Eacpp {

template <typename DecisionVariableType>
class MpMoeadIdealTopology : public MpMoead<DecisionVariableType> {
   private:
    std::string _idealTopologyFilePath;
    std::vector<int> _idealTopologyToSend;

   public:
    MpMoeadIdealTopology(
        int generationNum, int neighborhoodSize, int divisionsNumOfWeightVector,
        int migrationInterval, std::string idealTopologyFilePath,
        const std::shared_ptr<ICrossover<DecisionVariableType>>& crossover,
        const std::shared_ptr<IDecomposition>& decomposition,
        const std::shared_ptr<IMutation<DecisionVariableType>>& mutation,
        const std::shared_ptr<IProblem<DecisionVariableType>>& problem,
        const std::shared_ptr<IRepair<DecisionVariableType>>& repair,
        const std::shared_ptr<ISampling<DecisionVariableType>>& sampling,
        const std::shared_ptr<ISelection>& selection, bool isAsync = true)
        : MpMoead<DecisionVariableType>(
              generationNum, neighborhoodSize, divisionsNumOfWeightVector,
              migrationInterval, crossover, decomposition, mutation, problem,
              repair, sampling, selection, isAsync),
          _idealTopologyFilePath(idealTopologyFilePath) {}

   private:
    /// @brief 初期化時に、理想点の通信トポロジーを読み込む
    void AdditionalInitialization() override {
        std::ifstream ifs = OpenInputFile(_idealTopologyFilePath);
        std::string line;
        // rank行目をlineに読み込む
        for (int i = 0; i <= this->_rank; ++i) {
            if (!std::getline(ifs, line)) {
                throw std::runtime_error(
                    "Not enough lines in idealTopologyFile");
            }
        }

        std::stringstream ss(line);
        std::string item;
        while (std::getline(ss, item, ',')) {
            int rank = std::stoi(item);
            _idealTopologyToSend.push_back(rank);
            if (std::ranges::find(this->_neighboringRanks, rank) ==
                this->_neighboringRanks.end()) {
                this->_neighboringRanks.push_back(rank);
            }
        }
    }

    void AdditionalClear() override {
        _idealTopologyToSend.clear();
    }

    void PushIdealPointToSend(
        std::unordered_map<int, std::vector<double>>& dataToSend) override {
        if (this->_isIdealPointUpdated) {
            for (auto&& rank : _idealTopologyToSend) {
                dataToSend[rank].insert(
                    dataToSend[rank].end(),
                    this->_decomposition->IdealPoint().begin(),
                    this->_decomposition->IdealPoint().end());
            }
        } else if (!this->_isAsync) {
            for (auto&& rank : _idealTopologyToSend) {
                dataToSend.try_emplace(rank);
            }
        }
    }
};

}  // namespace Eacpp
