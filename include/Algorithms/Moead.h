#pragma once

#include <algorithm>
#include <eigen3/Eigen/Core>
#include <memory>
#include <numeric>
#include <ranges>
#include <tuple>
#include <vector>

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
class Moead {
   public:
    int generationNum;
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

    Moead(int generationNum, int decisionVariablesNum, int objectivesNum, int neighborhoodSize, int divisionsNumOfWeightVector,
          std::shared_ptr<ICrossover<DecisionVariableType>> crossover, std::shared_ptr<IDecomposition> decomposition,
          std::shared_ptr<IMutation<DecisionVariableType>> mutation, std::shared_ptr<IProblem<DecisionVariableType>> problem,
          std::shared_ptr<IRepair<DecisionVariableType>> repair, std::shared_ptr<ISampling<DecisionVariableType>> sampling,
          std::shared_ptr<ISelection> selection)
        : generationNum(generationNum),
          decisionVariablesNum(decisionVariablesNum),
          objectivesNum(objectivesNum),
          neighborhoodSize(neighborhoodSize),
          divisionsNumOfWeightVector(divisionsNumOfWeightVector),
          crossover(crossover),
          decomposition(decomposition),
          mutation(mutation),
          problem(problem),
          repair(repair),
          sampling(sampling),
          selection(selection) {}
    virtual ~Moead() {}

    void Initialize();
    void Update();
    void Run();
    std::vector<Eigen::ArrayXd> GetObjectivesList() const;

   private:
    std::vector<Individual<DecisionVariableType>> individuals;

    void CalculatePopulationNum();
    void InitializeWeightVectors();
    void InitializeNeighborhoods();
    void InitializeIndividuals();
    void InitializeIdealPoint();
    Individual<DecisionVariableType> GenerateNewIndividual(Individual<DecisionVariableType>& individual);
    void UpdateNeighborhood(Individual<DecisionVariableType>& individual, Individual<DecisionVariableType>& newIndividual);
};

template <typename DecisionVariableType>
void Moead<DecisionVariableType>::Initialize() {
    CalculatePopulationNum();
    InitializeIndividuals();
    InitializeIdealPoint();
    InitializeWeightVectors();
    InitializeNeighborhoods();
}

template <typename DecisionVariableType>
void Moead<DecisionVariableType>::Update() {
    for (auto&& individual : individuals) {
        Individual<DecisionVariableType> newIndividual = GenerateNewIndividual(individual);
        if (!problem->IsFeasible(newIndividual)) {
            repair->Repair(newIndividual);
        }
        problem->ComputeObjectiveSet(newIndividual);
        decomposition->UpdateIdealPoint(newIndividual.objectives);
        UpdateNeighborhood(individual, newIndividual);
    }
}

template <typename DecisionVariableType>
void Moead<DecisionVariableType>::Run() {
    Initialize();
    for (int i = 0; i < generationNum; i++) {
        Update();
    }
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
void Moead<DecisionVariableType>::CalculatePopulationNum() {
    int n = divisionsNumOfWeightVector + objectivesNum - 1;
    int r = objectivesNum - 1;
    populationSize = Combination(n, r);
}

template <typename DecisionVariableType>
void Moead<DecisionVariableType>::InitializeWeightVectors() {
    std::vector<double> numeratorOfWeightVector(divisionsNumOfWeightVector + 1);
    std::iota(numeratorOfWeightVector.begin(), numeratorOfWeightVector.end(), 0);
    std::vector<std::vector<double>> product = Product(numeratorOfWeightVector, objectivesNum);
    product.erase(std::remove_if(product.begin(), product.end(),
                                 [&](auto v) { return std::reduce(v.begin(), v.end()) != divisionsNumOfWeightVector; }),
                  product.end());

    if (product.size() != individuals.size()) {
        throw std::invalid_argument("The number of weight vectors is not equal to the number of individuals");
    }

    for (int i = 0; i < product.size(); i++) {
        individuals[i].weightVector = Eigen::Map<Eigen::ArrayXd>(product[i].data(), product[i].size());
        individuals[i].weightVector /= divisionsNumOfWeightVector;
    }
}

template <typename DecisionVariableType>
void Moead<DecisionVariableType>::InitializeNeighborhoods() {
    std::vector<std::vector<std::pair<double, int>>> euclideanDistances(populationSize,
                                                                        std::vector<std::pair<double, int>>(populationSize));
    for (int i = 0; i < populationSize; i++) {
        for (int j = 0; j < populationSize; j++) {
            euclideanDistances[i][j] =
                std::make_pair(individuals[i].CalculateSquaredEuclideanDistanceOfWeightVector(individuals[j]), j);
        }
    }
    for (int i = 0; i < populationSize; i++) {
        std::sort(euclideanDistances[i].begin(), euclideanDistances[i].end());
        for (int j = 0; j < neighborhoodSize; j++) {
            individuals[i].neighborhood.push_back(euclideanDistances[i][j].second);
        }
    }
}

template <typename DecisionVariableType>
void Moead<DecisionVariableType>::InitializeIndividuals() {
    individuals = sampling->Sample(populationSize, decisionVariablesNum);
    for (auto&& individual : individuals) {
        problem->ComputeObjectiveSet(individual);
    }
}

template <typename DecisionVariableType>
void Moead<DecisionVariableType>::InitializeIdealPoint() {
    for (auto&& individual : individuals) {
        decomposition->UpdateIdealPoint(individual.objectives);
    }
}

template <typename DecisionVariableType>
Individual<DecisionVariableType> Moead<DecisionVariableType>::GenerateNewIndividual(
    Individual<DecisionVariableType>& individual) {
    std::vector<int> childrenIndex = selection->Select(crossover->GetParentNum(), individual.neighborhood);
    std::vector<Individual<DecisionVariableType>> parents;
    for (const auto& i : childrenIndex) {
        parents.push_back(individuals[i]);
    }
    Individual<DecisionVariableType> newIndividual = crossover->Cross(parents);
    mutation->Mutate(newIndividual);
    return newIndividual;
}

template <typename DecisionVariableType>
void Moead<DecisionVariableType>::UpdateNeighborhood(Individual<DecisionVariableType>& individual,
                                                     Individual<DecisionVariableType>& newIndividual) {
    for (auto&& i : individual.neighborhood) {
        double newSubObjective = decomposition->ComputeObjective(individuals[i].weightVector, newIndividual.objectives);
        double oldSubObjective = decomposition->ComputeObjective(individuals[i].weightVector, individuals[i].objectives);
        if (newSubObjective <= oldSubObjective) {
            individuals[i].UpdateFrom(newIndividual);
        }
    }
}

}  // namespace Eacpp
