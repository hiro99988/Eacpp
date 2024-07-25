#ifndef Moead_H
#define Moead_H

#include <algorithm>
#include <eigen3/Eigen/Core>
#include <memory>
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
    std::shared_ptr<ISelection<DecisionVariableType>> selection;

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
    void GenerateWeightVectors() {
        std::vector<double> takedSetTop(H + 1);
        std::iota(takedSetTop.begin(), takedSetTop.end(), 0);
        auto product = Product(takedSetTop, objectiveNum);
        product.erase(std::remove_if(product.begin(), product.end(),
                                     [&](std::vector<int> v) { return std::reduce(v.begin(), v.end()) != H; }),
                      product.end());

        //     Eigen::ArrayXXd product = Eigen::ArrayXXd::Zero(std::pow(H + 1, decisionVariableNum), decisionVariableNum);
        // for (int i = 0; i < decisionVariableNum; i++) {
        //     product.col(i) = takedSetTop;
        // }
        // Eigen::ArrayXXd wvTop = product.rowwise().sum();
        // weightVectors = wvTop.rowwise() / H;
    }

    // def generate_weight_vector(m, H):
    //     taked_set_top = [i for i in range(H + 1)]
    //     product = np.array(list(itertools.product(taked_set_top, repeat=m)))
    //     wv_top = product[np.sum(product, axis=1) == H]
    //     wv = wv_top / H
    //     return wv

    void GenerateNeighborhoods() {
        Eigen::ArrayXd euclideanDistances(populationSize, populationSize);
        for (int i = 0; i < populationSize; i++) {
            euclideanDistances.col(i) = (weightVectors.colwise() - weightVectors.col(i)).colwise().squaredNorm();
        }
        neighborhoodIndexes.resize(neighborNum, populationSize);
        for (int i = 0; i < populationSize; i++) {
            std::sort(euclideanDistances.col(i).begin(), euclideanDistances.col(i).end());
            neighborhoodIndexes.col(i) = euclideanDistances.col(i).head(neighborNum);
        }
    }

    void InitializePopulation() {
        solutions = sampling->Sample(populationSize, decisionVariableNum);
        objectiveSets.resize(objectiveNum, populationSize);
        for (int i = 0; i < populationSize; i++) {
            objectiveSets.col(i) = problem->ComputeObjectiveSet(solutions.row(i));
        }
    }

    void InitializeIdealPoint() { idealPoint = objectiveSets.rowwise().minCoeff(); }

    Eigen::ArrayX<DecisionVariableType> GenerateNewSolution(int index) {
        Eigen::ArrayXi childrenIndex = selection->Select(crossover->GetParentNum(), neighborhoodIndexes.col(index));
        Eigen::ArrayXX<DecisionVariableType> parents = solutions.col(childrenIndex);
        Eigen::ArrayX<DecisionVariableType> newSolution = crossover->Cross(parents);
        mutation->Mutate(newSolution);
        return newSolution;
    }

    void UpdateIdealPoint(Eigen::ArrayXd objectiveSet) { idealPoint = idealPoint.min(objectiveSet); }

    void UpdateNeighboringSolutions(int index, Eigen::ArrayX<DecisionVariableType> solution, Eigen::ArrayXd objectiveSet) {
        for (auto&& i : neighborhoodIndexes.col(index)) {
            double newSubObjective = decomposition->ComputeObjective(weightVectors.col(i), objectiveSet, idealPoint);
            double oldSubObjective = decomposition->ComputeObjective(weightVectors.col(i), objectiveSets.col(i), idealPoint);
            if (newSubObjective <= oldSubObjective) {
                solutions.col(i) = solution;
                objectiveSets.col(i) = objectiveSet;
            }
        }
    }
};

}  // namespace Eacpp

#endif
