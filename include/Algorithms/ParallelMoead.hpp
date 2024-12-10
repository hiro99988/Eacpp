#pragma once

#include <mpi.h>

#include <algorithm>
#include <array>
#include <eigen3/Eigen/Core>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <ranges>
#include <set>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "Algorithms/IMoead.h"
#include "Algorithms/MoeadInitializer.h"
#include "Crossovers/ICrossover.h"
#include "Decompositions/IDecomposition.h"
#include "Individual.h"
#include "Mutations/IMutation.h"
#include "Problems/IProblem.h"
#include "Repairs/IRepair.h"
#include "Samplings/ISampling.h"
#include "Selections/ISelection.h"
#include "Utils/MpiUtils.h"
#include "Utils/Utils.h"

namespace Eacpp {

template <typename DecisionVariableType>
class ParallelMoead : public IMoead<DecisionVariableType> {
   public:
    constexpr static int maxBufferSize = 100;
    constexpr static int messageTag = 0;

    ParallelMoead(int generationNum, int neighborhoodSize, int divisionsNumOfWeightVector, int migrationInterval,
                  bool idealPointMigration, const std::shared_ptr<ICrossover<DecisionVariableType>>& crossover,
                  const std::shared_ptr<IDecomposition>& decomposition,
                  const std::shared_ptr<IMutation<DecisionVariableType>>& mutation,
                  const std::shared_ptr<IProblem<DecisionVariableType>>& problem,
                  const std::shared_ptr<IRepair<DecisionVariableType>>& repair,
                  const std::shared_ptr<ISampling<DecisionVariableType>>& sampling,
                  const std::shared_ptr<ISelection>& selection)
        : generationNum(generationNum),
          neighborhoodSize(neighborhoodSize),
          divisionsNumOfWeightVector(divisionsNumOfWeightVector),
          migrationInterval(migrationInterval),
          idealPointMigration(idealPointMigration) {
        if (!crossover || !decomposition || !mutation || !problem || !repair || !sampling || !selection) {
            throw std::invalid_argument("Null pointer is passed");
        }

        this->crossover = crossover;
        this->decomposition = decomposition;
        this->mutation = mutation;
        this->problem = problem;
        this->repair = repair;
        this->sampling = sampling;
        this->selection = selection;

        decisionVariablesNum = problem->DecisionVariablesNum();
        objectivesNum = problem->ObjectivesNum();
        currentGeneration = 0;
        singleMessageSize = decisionVariablesNum + objectivesNum + 1;
        isIdealPointUpdated = false;
    }
    virtual ~ParallelMoead() {}

    int CurrentGeneration() const override;
    void Run() override;
    void Initialize() override;
    void Update() override;
    bool IsEnd() const override;
    std::vector<Eigen::ArrayXd> GetObjectivesList() const override;
    std::vector<Eigen::ArrayX<DecisionVariableType>> GetSolutionList() const override;

   private:
    int totalPopulationSize;
    int generationNum;
    int currentGeneration;
    int decisionVariablesNum;
    int objectivesNum;
    int neighborhoodSize;
    int migrationInterval;
    int divisionsNumOfWeightVector;
    std::shared_ptr<ICrossover<DecisionVariableType>> crossover;
    std::shared_ptr<IDecomposition> decomposition;
    std::shared_ptr<IMutation<DecisionVariableType>> mutation;
    std::shared_ptr<IProblem<DecisionVariableType>> problem;
    std::shared_ptr<IRepair<DecisionVariableType>> repair;
    std::shared_ptr<ISampling<DecisionVariableType>> sampling;
    std::shared_ptr<ISelection> selection;
    int rank;
    int parallelSize;
    MoeadInitializer initializer;
    std::vector<int> internalIndexes;
    std::vector<int> externalIndexes;
    std::set<int> updatedSolutionIndexes;
    std::unordered_map<int, Individual<DecisionVariableType>> individuals;
    std::unordered_map<int, Individual<DecisionVariableType>> clonedExternalIndividuals;
    std::vector<int> ranksForExternalIndividuals;
    std::set<int> neighboringRanks;
    std::vector<int> ranksToSend;
    int singleMessageSize;
    bool idealPointMigration;
    bool isIdealPointUpdated;

    void InitializeMpi();
    void InitializeIsland();
    void Clear();
    std::pair<std::vector<int>, std::vector<int>> GenerateExternalNeighborhood(std::vector<int>& neighborhoodIndexes,
                                                                               std::vector<int>& populationSizes);
    std::vector<double> GetWeightVectorsMatchingIndexes(std::vector<double>& weightVectors, std::vector<int>& indexes);
    void CalculateNeighboringRanks();
    void InitializeIndividualAndWeightVector(std::vector<Eigen::ArrayXd>& weightVectors,
                                             std::vector<std::vector<int>>& neighborhoodIndexes,
                                             std::vector<Eigen::ArrayXd>& externalNeighboringWeightVectors);
    void CalculateRanksToSent(const std::vector<int>& neighborhoodIndexes, const std::vector<int>& populationSizes,
                              std::vector<int>& outRanksToSentByRank, std::vector<int>& outSizes);
    std::vector<std::vector<double>> ScatterPopulation();

    void InitializePopulation();
    void InitializeExternalPopulation(std::vector<std::vector<double>>& receivedSolutions);
    void MakeLocalCopyOfExternalIndividuals();
    std::vector<Individual<DecisionVariableType>> SelectParents(int index);
    Individual<DecisionVariableType> GenerateNewIndividual(int index);
    void UpdateNeighboringIndividuals(int index, Individual<DecisionVariableType>& newIndividual);
    bool IsInternal(int index);
    bool IsExternal(int index);
    void UpdateIdealPoint(const Eigen::ArrayXd& objectives);
    void UpdateIdealPointWithMessage(const std::vector<double>& message);

    std::unordered_map<int, std::vector<double>> CreateMessages();
    void SendMessages();
    std::vector<std::vector<double>> ReceiveMessages();
    void UpdateWithMessage(std::vector<double>& message);

#ifdef _TEST_
   public:
    friend class MpMoeadTest;
    friend class MpMoeadTestM;
#endif
};

}  // namespace Eacpp