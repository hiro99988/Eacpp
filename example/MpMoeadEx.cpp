#include <mpi.h>

#include <fstream>
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
    int rank;
    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int generationNum = 100;
    int neighborNum = 3;
    int migrationInterval = 1;
    int H = 99;

    if (argc == 5) {
        generationNum = std::stoi(argv[1]);
        neighborNum = std::stoi(argv[2]);
        migrationInterval = std::stoi(argv[3]);
        H = std::stoi(argv[4]);
    }

    int totalPopulationSize = H + 1;

    auto problem = std::make_shared<ZDT1>();

    int decisionVariableNum = problem->decisionVariablesNum;
    int objectiveNum = problem->objectivesNum;
    std::vector<std::array<double, 2>> variableBounds = {problem->variableBound};

    auto crossover = std::make_shared<BinomialCrossover>(1.0, 0.5);
    auto decomposition = std::make_shared<Tchebycheff>();
    auto mutation = std::make_shared<PolynomialMutation>(1.0 / decisionVariableNum, 20.0, variableBounds);
    auto sampling = std::make_shared<UniformRandomSampling>(problem->variableBound[0], problem->variableBound[1]);
    auto selection = std::make_shared<RandomSelection>();

    MpMoead<double> moead(totalPopulationSize, generationNum, decisionVariableNum, objectiveNum, neighborNum, migrationInterval,
                          H, crossover, decomposition, mutation, problem, sampling, selection);

    moead.Run();

    auto allObjectives = moead.GetAllObjectives();

    if (rank == 0) {
        std::ofstream ofs("out/data/result.txt");
        for (const auto& set : allObjectives) {
            for (const auto& value : set) {
                ofs << value << " ";
            }
            ofs << std::endl;
        }
    }

    MPI_Finalize();

    return 0;
}