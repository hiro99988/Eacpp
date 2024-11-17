#pragma once

#include <algorithm>
#include <eigen3/Eigen/Core>
#include <memory>
#include <numeric>
#include <ranges>
#include <tuple>
#include <vector>

#include "Algorithms/IMoead.h"
#include "Algorithms/MoeadInitializer.h"
#include "Crossovers/ICrossover.h"
#include "Decompositions/IDecomposition.h"
#include "Individual/Individual.h"
#include "Mutations/IMutation.h"
#include "Problems/IProblem.h"
#include "Repairs/IRepair.h"
#include "Samplings/ISampling.h"
#include "Selections/ISelection.h"
#include "Utils/Utils.h"

namespace Eacpp {

template <typename DecisionVariableType>
class Moead : public IMoead<DecisionVariableType> {
   public:
    Moead(int generationNum, int neighborhoodSize, int divisionsNumOfWeightVector,
          const std::shared_ptr<ICrossover<DecisionVariableType>>& crossover,
          const std::shared_ptr<IDecomposition>& decomposition,
          const std::shared_ptr<IMutation<DecisionVariableType>>& mutation,
          const std::shared_ptr<IProblem<DecisionVariableType>>& problem,
          const std::shared_ptr<IRepair<DecisionVariableType>>& repair,
          const std::shared_ptr<ISampling<DecisionVariableType>>& sampling, const std::shared_ptr<ISelection>& selection)
        : generationNum(generationNum),
          neighborhoodSize(neighborhoodSize),
          divisionsNumOfWeightVector(divisionsNumOfWeightVector) {
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
    }

    virtual ~Moead() {}

    int GenerationNum() const override {
        return generationNum;
    }
    void Initialize() override;
    void Update() override;
    void Run() override;
    bool IsEnd() const override;
    std::vector<Eigen::ArrayXd> GetObjectivesList() const override;
    std::vector<Eigen::ArrayX<DecisionVariableType>> GetSolutionList() const override;

   private:
    int generationNum;
    int currentGeneration;
    int populationSize;
    int decisionVariablesNum;
    int objectivesNum;
    int neighborhoodSize;
    int divisionsNumOfWeightVector;
    std::shared_ptr<ICrossover<DecisionVariableType>> crossover;
    std::shared_ptr<IDecomposition> decomposition;
    std::shared_ptr<IMutation<DecisionVariableType>> mutation;
    std::shared_ptr<IProblem<DecisionVariableType>> problem;
    std::shared_ptr<IRepair<DecisionVariableType>> repair;
    std::shared_ptr<ISampling<DecisionVariableType>> sampling;
    std::shared_ptr<ISelection> selection;
    std::vector<Individual<DecisionVariableType>> individuals;
    MoeadInitializer initializer;

    void InitializeIndividuals(const std::vector<Eigen::ArrayXd>& weightVectors,
                               const std::vector<std::vector<int>>& neighborhoods);
    void InitializeIdealPoint();
    Individual<DecisionVariableType> GenerateNewIndividual(const Individual<DecisionVariableType>& individual);
    void UpdateNeighborhood(const Individual<DecisionVariableType>& individual,
                            const Individual<DecisionVariableType>& newIndividual);
};

template <typename DecisionVariableType>
void Moead<DecisionVariableType>::Initialize() {
    populationSize = initializer.CalculatePopulationSize(divisionsNumOfWeightVector, objectivesNum);

    std::vector<Eigen::ArrayXd> weightVectors;
    std::vector<std::vector<int>> neighborhoods;
    initializer.GenerateWeightVectorsAndNeighborhoods(divisionsNumOfWeightVector, objectivesNum, neighborhoodSize,
                                                      weightVectors, neighborhoods);

    InitializeIndividuals(weightVectors, neighborhoods);
    InitializeIdealPoint();

    currentGeneration = 0;
}

template <typename DecisionVariableType>
void Moead<DecisionVariableType>::Update() {
    for (auto&& individual : individuals) {
        Individual<DecisionVariableType> newIndividual = GenerateNewIndividual(individual);
        repair->Repair(newIndividual);
        problem->ComputeObjectiveSet(newIndividual);
        decomposition->UpdateIdealPoint(newIndividual.objectives);
        UpdateNeighborhood(individual, newIndividual);
    }

    currentGeneration++;
}

template <typename DecisionVariableType>
void Moead<DecisionVariableType>::Run() {
    Initialize();
    while (!IsEnd()) {
        Update();
    }
}

template <typename DecisionVariableType>
bool Moead<DecisionVariableType>::IsEnd() const {
    return currentGeneration >= generationNum;
}

template <typename DecisionVariableType>
std::vector<Eigen::ArrayXd> Moead<DecisionVariableType>::GetObjectivesList() const {
    std::vector<Eigen::ArrayXd> objectives;
    for (const auto& individual : individuals) {
        objectives.push_back(individual.objectives);
    }
    return objectives;
}

template <typename DecisionVariableType>
std::vector<Eigen::ArrayX<DecisionVariableType>> Moead<DecisionVariableType>::GetSolutionList() const {
    std::vector<Eigen::ArrayX<DecisionVariableType>> solutions;
    for (const auto& individual : individuals) {
        solutions.push_back(individual.solution);
    }
    return solutions;
}

template <typename DecisionVariableType>
void Moead<DecisionVariableType>::InitializeIndividuals(const std::vector<Eigen::ArrayXd>& weightVectors,
                                                        const std::vector<std::vector<int>>& neighborhoods) {
    individuals = sampling->Sample(populationSize, decisionVariablesNum);
    for (int i = 0; i < populationSize; i++) {
        individuals[i].weightVector = std::move(weightVectors[i]);
        individuals[i].neighborhood = std::move(neighborhoods[i]);
        problem->ComputeObjectiveSet(individuals[i]);
    }
}

template <typename DecisionVariableType>
void Moead<DecisionVariableType>::InitializeIdealPoint() {
    decomposition->InitializeIdealPoint(objectivesNum);
    for (auto&& individual : individuals) {
        decomposition->UpdateIdealPoint(individual.objectives);
    }
}

template <typename DecisionVariableType>
Individual<DecisionVariableType> Moead<DecisionVariableType>::GenerateNewIndividual(
    const Individual<DecisionVariableType>& individual) {
    std::vector<int> childrenIndex = selection->Select(crossover->GetParentNum(), individual.neighborhood);
    std::vector<Individual<DecisionVariableType>> parents;
    parents.reserve(childrenIndex.size());
    for (const auto& i : childrenIndex) {
        parents.push_back(individuals[i]);
    }
    Individual<DecisionVariableType> newIndividual = crossover->Cross(parents);
    mutation->Mutate(newIndividual);
    return newIndividual;
}

template <typename DecisionVariableType>
void Moead<DecisionVariableType>::UpdateNeighborhood(const Individual<DecisionVariableType>& individual,
                                                     const Individual<DecisionVariableType>& newIndividual) {
    for (auto&& i : individual.neighborhood) {
        double newSubObjective = decomposition->ComputeObjective(individuals[i].weightVector, newIndividual.objectives);
        double oldSubObjective = decomposition->ComputeObjective(individuals[i].weightVector, individuals[i].objectives);
        if (newSubObjective <= oldSubObjective) {
            individuals[i].UpdateFrom(newIndividual);
        }
    }
}

}  // namespace Eacpp
