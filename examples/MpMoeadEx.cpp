#include <mpi.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <tuple>
#include <vector>

#include "Algorithms/MpMoead.h"
#include "Crossovers/BinomialCrossover.h"
#include "Decompositions/Tchebycheff.h"
#include "Mutations/PolynomialMutation.h"
#include "Problems/ZDT1.h"
#include "Repairs/SamplingRepair.h"
#include "Samplings/UniformRandomSampling.h"
#include "Selections/RandomSelection.h"

using namespace Eacpp;

int main(int argc, char** argv) {
    int rank;
    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int generationNum = 100;
    int neighborhoodSize = 5;
    int migrationInterval = 1;
    int H = 99;

    if (argc == 5) {
        generationNum = std::stoi(argv[1]);
        neighborhoodSize = std::stoi(argv[2]);
        migrationInterval = std::stoi(argv[3]);
        H = std::stoi(argv[4]);
    }

    int totalPopulationSize = H + 1;

    auto problem = std::make_shared<ZDT1>();

    int decisionVariableNum = problem->DecisionVariablesNum();
    int objectivesNum = problem->ObjectivesNum();
    std::pair<double, double> variableBound{problem->VariableBound()};
    std::vector<std::pair<double, double>> variableBounds{variableBound};

    auto crossover = std::make_shared<BinomialCrossover>(1.0, 0.5);
    auto decomposition = std::make_shared<Tchebycheff>(objectivesNum);
    auto mutation = std::make_shared<PolynomialMutation>(1.0 / decisionVariableNum, 20.0, variableBounds);
    auto sampling = std::make_shared<UniformRandomSampling>(variableBound.first, variableBound.second);
    auto repair = std::make_shared<SamplingRepair<double>>(sampling);
    auto selection = std::make_shared<RandomSelection>();

    MpMoead<double> moead(totalPopulationSize, generationNum, decisionVariableNum, objectivesNum, neighborhoodSize,
                          migrationInterval, H, crossover, decomposition, mutation, problem, repair, sampling, selection);

    double start = MPI_Wtime();

    moead.Run();

    double end = MPI_Wtime();
    double executionTime = end - start;
    double maxTime;
    MPI_Reduce(&executionTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "Maximum execution time across all processes: " << maxTime << " seconds" << std::endl;
    }

    // moead.WriteAllObjectives();

    // moead.WriteTransitionOfIdealPoint();

    MPI_Finalize();

    return 0;
}