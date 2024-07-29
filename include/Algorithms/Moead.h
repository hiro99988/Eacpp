#ifndef Moead_H
#define Moead_H

#include <algorithm>
#include <eigen3/Eigen/Dense>
#include <memory>
#include <numeric>
#include <ranges>
#include <tuple>
#include <vector>

#include "Crossovers/ICrossover.h"
#include "Decompositions/IDecomposition.h"
#include "Mutations/IMutation.h"
#include "Problems/IProblem.h"
#include "Samplings/ISampling.h"
#include "Selections/ISelection.h"
#include "Utils/TemplateType.h"
#include "Utils/Utils.h"

namespace Eacpp {

template <Number DecisionVariableType>
class Moead {
   public:
    int generationNum;
    int populationSize;
    int decisionVariableNum;
    int objectiveNum;
    int neighborNum;
    int H;
    std::shared_ptr<ICrossover<DecisionVariableType>> crossover;
    std::shared_ptr<IDecomposition> decomposition;
    std::shared_ptr<IMutation<DecisionVariableType>> mutation;
    std::shared_ptr<IProblem<DecisionVariableType>> problem;
    std::shared_ptr<ISampling<DecisionVariableType>> sampling;
    std::shared_ptr<ISelection> selection;
    std::vector<Eigen::ArrayXd> weightVectors;
    std::vector<Eigen::ArrayX<DecisionVariableType>> solutions;
    std::vector<Eigen::ArrayXd> objectiveSets;
    Eigen::ArrayXd idealPoint;
    std::vector<std::vector<int>> neighborhoodIndexes;

    Moead(int generationNum, int decisionVariableNum, int objectiveNum, int neighborNum, int H,
          std::shared_ptr<ICrossover<DecisionVariableType>> crossover, std::shared_ptr<IDecomposition> decomposition,
          std::shared_ptr<IMutation<DecisionVariableType>> mutation, std::shared_ptr<IProblem<DecisionVariableType>> problem,
          std::shared_ptr<ISampling<DecisionVariableType>> sampling, std::shared_ptr<ISelection> selection)
        : generationNum(generationNum),
          decisionVariableNum(decisionVariableNum),
          objectiveNum(objectiveNum),
          neighborNum(neighborNum),
          H(H),
          crossover(crossover),
          decomposition(decomposition),
          mutation(mutation),
          problem(problem),
          sampling(sampling),
          selection(selection) {}
    virtual ~Moead() {}

    void Initialize() {
        CaluculatePopulationNum();
        GenerateWeightVectors();
        GenerateNeighborhoods();
        InitializePopulation();
        InitializeIdealPoint();
    }

    void Update() {
        for (int i = 0; i < populationSize; i++) {
            Eigen::ArrayX<DecisionVariableType> newSolution = GenerateNewSolution(i);
            if (!problem->IsFeasible(newSolution)) {
                newSolution = sampling->Sample(1, decisionVariableNum)[0];
            }
            Eigen::ArrayXd newObjectiveSet = problem->ComputeObjectiveSet(newSolution);
            UpdateIdealPoint(newObjectiveSet);
            UpdateNeighboringSolutions(i, newSolution, newObjectiveSet);
        }
    }

    void Run() {
        Initialize();
        for (int i = 0; i < generationNum; i++) {
            Update();
        }
    }

   private:
    void CaluculatePopulationNum() {
        int n = H + objectiveNum - 1;
        int r = objectiveNum - 1;
        populationSize = Combination(n, r);
    }

    void GenerateWeightVectors() {
        std::vector<double> takedSetTop(H + 1);
        std::iota(takedSetTop.begin(), takedSetTop.end(), 0);
        std::vector<std::vector<double>> product = Product(takedSetTop, objectiveNum);
        product.erase(
            std::remove_if(product.begin(), product.end(), [&](auto v) { return std::reduce(v.begin(), v.end()) != H; }),
            product.end());
        for (auto&& v : product) {
            weightVectors.push_back(Eigen::Map<Eigen::ArrayXd>(v.data(), v.size()));
        }
    }

    void GenerateNeighborhoods() {
        std::vector<std::vector<std::pair<double, int>>> euclideanDistances(
            populationSize, std::vector<std::pair<double, int>>(populationSize));
        for (int i = 0; i < populationSize; i++) {
            for (int j = 0; j < populationSize; j++) {
                euclideanDistances[i][j] = std::make_pair((weightVectors[i] - weightVectors[j]).matrix().squaredNorm(), j);
            }
        }
        neighborhoodIndexes = std::vector<std::vector<int>>(populationSize, std::vector<int>(neighborNum));
        for (int i = 0; i < populationSize; i++) {
            std::sort(euclideanDistances[i].begin(), euclideanDistances[i].end());
            for (int j = 0; j < neighborNum; j++) {
                neighborhoodIndexes[i][j] = euclideanDistances[i][j].second;
            }
        }
    }

    void InitializePopulation() {
        solutions = sampling->Sample(populationSize, decisionVariableNum);
        objectiveSets.reserve(populationSize);
        for (const auto& solution : solutions) {
            objectiveSets.push_back(problem->ComputeObjectiveSet(solution));
        }
    }

    void InitializeIdealPoint() {
        Eigen::ArrayXd tmp = objectiveSets[0];
        for (int i = 1; i < populationSize; i++) {
            tmp = tmp.min(objectiveSets[i]);
        }
        idealPoint = tmp;
    }

    Eigen::ArrayX<DecisionVariableType> GenerateNewSolution(int index) {
        std::vector<int> childrenIndex = selection->Select(crossover->GetParentNum(), neighborhoodIndexes[index]);
        std::vector<Eigen::ArrayX<DecisionVariableType>> parents;
        for (const auto& i : childrenIndex) {
            parents.push_back(solutions[i]);
        }
        Eigen::ArrayX<DecisionVariableType> newSolution = crossover->Cross(parents);
        mutation->Mutate(newSolution);
        return newSolution;
    }

    void UpdateIdealPoint(Eigen::ArrayXd objectiveSet) { idealPoint = idealPoint.min(objectiveSet); }

    void UpdateNeighboringSolutions(int index, Eigen::ArrayX<DecisionVariableType> solution, Eigen::ArrayXd objectiveSet) {
        for (auto&& i : neighborhoodIndexes[index]) {
            double newSubObjective = decomposition->ComputeObjective(weightVectors[i], objectiveSet, idealPoint);
            double oldSubObjective = decomposition->ComputeObjective(weightVectors[i], objectiveSets[i], idealPoint);
            if (newSubObjective <= oldSubObjective) {
                solutions[i] = solution;
                objectiveSets[i] = objectiveSet;
            }
        }
    }
};

}  // namespace Eacpp

#endif
