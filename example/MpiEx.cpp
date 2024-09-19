#include <mpi.h>

#include <iostream>

#include "Algorithms/MpMoead.h"
#include "Crossovers/BinomialCrossover.h"
#include "Decompositions/Tchebycheff.h"
#include "Mutations/PolynomialMutation.h"
#include "Problems/ZDT1.h"
#include "Samplings/UniformRandomSampling.h"
#include "Selections/RandomSelection.h"

using namespace Eacpp;

int main(int argc, char** argv) {
    int generationNum = 300;
    int H = 99;
    int totalPopulationSize = H + 1;
    int neighborNum = 3;
    int migrationInterval = 1;

    std::shared_ptr<ZDT1> problem = std::make_shared<ZDT1>();

    int decisionVariableNum = problem->decisionNum;
    int objectiveNum = problem->objectiveNum;
    std::vector<std::array<double, 2>> variableBounds = {problem->variableBound};

    std::shared_ptr<BinomialCrossover> crossover = std::make_shared<BinomialCrossover>(1.0, 0.5);
    std::shared_ptr<Tchebycheff> decomposition = std::make_shared<Tchebycheff>();
    std::shared_ptr<PolynomialMutation> mutation =
        std::make_shared<PolynomialMutation>(1.0 / decisionVariableNum, 20.0, variableBounds);
    std::shared_ptr<UniformRandomSampling> sampling =
        std::make_shared<UniformRandomSampling>(problem->variableBound[0], problem->variableBound[1]);
    std::shared_ptr<RandomSelection> selection = std::make_shared<RandomSelection>();

    MpMoead<double> moead(totalPopulationSize, generationNum, decisionVariableNum, objectiveNum, neighborNum, migrationInterval,
                          H, crossover, decomposition, mutation, problem, sampling, selection);

    moead.Run();

    MPI_Finalize();

    return 0;
}