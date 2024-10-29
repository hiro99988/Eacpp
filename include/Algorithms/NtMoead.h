#pragma once

#include <algorithm>
#include <eigen3/Eigen/Core>
#include <memory>
#include <numeric>
#include <ranges>
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

    void Initialize() override;
    void Update() override;
    void Run() override;
    bool IsEnd() const override;
    std::vector<Eigen::ArrayXd> GetObjectivesList() const override;

   private:
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
    void UpdateNeighborhood(const Individual<DecisionVariableType>& individual,
                            const Individual<DecisionVariableType>& newIndividual);
    Individual<DecisionVariableType> GenerateNewIndividual(const Individual<DecisionVariableType>& individual);
};

}  // namespace Eacpp