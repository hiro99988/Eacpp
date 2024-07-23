#ifndef Moead_H
#define Moead_H

#include <eigen3/Eigen/Core>

#include "Crossovers/ICrossover.h"
#include "Decompositions/IDecomposition.h"
#include "Mutations/IMutation.h"
#include "Problems/IProblem.h"
#include "Samplings/ISampling.h"
#include "Selections/ISelection.h"
#include "Utils/TemplateType.h"

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
    ICrossover<DecisionVariableType>* crossover;
    IDecomposition* decomposition;
    IMutation<DecisionVariableType>* mutation;
    IProblem<DecisionVariableType>* problem;
    ISampling<DecisionVariableType>* sampling;
    ISelection<DecisionVariableType>* selection;

    Eigen::ArrayXXd weightVectors;
    Eigen::ArrayXX<DecisionVariableType> solutions;
    Eigen::ArrayXXd objectiveSets;
    Eigen::ArrayXd idealPoint;
    Eigen::ArrayXXi neighborhoodIndexes;

    Moead(int generationNum, int populationSize, int decisionVariableNum, int objectiveNum, int neighborNum, int H,
          ICrossover<DecisionVariableType>* crossover, IDecomposition* decomposition, IMutation<DecisionVariableType>* mutation,
          IProblem<DecisionVariableType>* problem, ISampling<DecisionVariableType>* sampling,
          ISelection<DecisionVariableType>* selection)
        : generationNum(generationNum),
          populationSize(populationSize),
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
        GenerateWeightVectors();
        GenerateNeighborhoods();
        InitializePopulation();
        InitializeIdealPoint();
    }

    void Update() {
        for (int i = 0; i < populationSize; i++) {
            Eigen::ArrayX<DecisionVariableType> newSolution = GenerateNewSolution(i);
            if (!problem->IsFeasible(newSolution)) {
                newSolution = sampling->Sample(1, decisionVariableNum);
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
    void GenerateWeightVectors();
    void GenerateNeighborhoods();
    void InitializePopulation();
    void InitializeIdealPoint();
    Eigen::ArrayX<DecisionVariableType> GenerateNewSolution(int index);
    void UpdateIdealPoint(Eigen::ArrayXd objectiveSet);
    void UpdateNeighboringSolutions(int index, Eigen::ArrayX<DecisionVariableType> solution, Eigen::ArrayXd objectiveSet);
};

}  // namespace Eacpp

#endif
