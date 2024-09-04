#pragma once

#include <mpi.h>

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

template <typename DecisionVariableType>
class MpMoead {
   public:
    int generationNum;
    int populationSize;
    int decisionVariableNum;
    int objectiveNum;
    int neighborNum;
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
    std::vector<Eigen::ArrayXd> allNeighborWeightVectors;
    std::vector<Eigen::ArrayX<DecisionVariableType>> allNeighborSolutions;
    std::vector<Eigen::ArrayXd> allNeighborObjectiveSets;
    std::vector<int> neighborWeightVectorIndexes;
    int rank;
    int parallelSize;

    MpMoead(int generationNum, int decisionVariableNum, int objectiveNum, int neighborNum,
            std::shared_ptr<ICrossover<DecisionVariableType>> crossover, std::shared_ptr<IDecomposition> decomposition,
            std::shared_ptr<IMutation<DecisionVariableType>> mutation, std::shared_ptr<IProblem<DecisionVariableType>> problem,
            std::shared_ptr<ISampling<DecisionVariableType>> sampling, std::shared_ptr<ISelection> selection)
        : generationNum(generationNum),
          decisionVariableNum(decisionVariableNum),
          objectiveNum(objectiveNum),
          neighborNum(neighborNum),
          crossover(crossover),
          decomposition(decomposition),
          mutation(mutation),
          problem(problem),
          sampling(sampling),
          selection(selection) {}
    virtual ~MpMoead() {}

    void Run(int argc, char** argv);
    void Initialize(int totalPopulationSize, int H);
    void InitializeIsland();
    void Update();

   private:
    void CaluculatePopulationNum(int totalPopulationSize);
    std::vector<double> GenerateAllWeightVectors(int H);
    std::vector<int> GenerateAllNeighborhoods(int totalPopulationSize, std::vector<double>& allWeightVectors);
    std::vector<int> GeneratePopulationSizes(int totalPopulationSize);
    std::vector<double> ScatterWeightVector(std::vector<double>& allWeightVectors, std::vector<int>& populationSizes);
    std::vector<int> ScatterNeighborhoodIndexes(std::vector<int>& allNeighborhoodIndexes, std::vector<int>& populationSizes);
    std::vector<double> SendNeighborWeightVectors(std::vector<double>& allWeightVectors,
                                                  std::vector<int>& allNeighborhoodIndexes, std::vector<int>& populationSizes);
    std::tuple<std::vector<int>, std::vector<int>> GenerateDataCountsAndDisplacements(std::vector<int>& populationSizes,
                                                                                      int dataSize);
    void ConvertWeightVectorsToEigenArrayXd(std::vector<double>& allWeightVectors);
    void ConvertNeighborhoodIndexesToVector2d(std::vector<int>& allNeighborhoods);
    void ConvertNeighborWeightVectorToEigenArrayXd(std::vector<double>& receivedNeighborWeightVectors);

    void InitializePopulation();
    void InitializeIdealPoint();
    Eigen::ArrayX<DecisionVariableType> GenerateNewSolution(int index);
    void UpdateIdealPoint(Eigen::ArrayXd objectiveSet);
    void UpdateNeighboringSolutions(int index, Eigen::ArrayX<DecisionVariableType> solution, Eigen::ArrayXd objectiveSet);

   public:
#ifdef _TEST_
    friend class MpMpeadTest;
#endif
};

template class MpMoead<int>;
template class MpMoead<float>;
template class MpMoead<double>;

}  // namespace Eacpp
