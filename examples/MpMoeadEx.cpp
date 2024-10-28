#include <mpi.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <tuple>
#include <vector>

#include "Algorithms/MpMoead.h"
#include "Crossovers/SimulatedBinaryCrossover.h"
#include "Decompositions/Tchebycheff.h"
#include "Mutations/PolynomialMutation.h"
#include "Problems/ZDT1.h"
#include "Repairs/SamplingRepair.h"
#include "Samplings/RealRandomSampling.h"
#include "Selections/RandomSelection.h"

using namespace Eacpp;

int main(int argc, char** argv) {
    int rank;
    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int generationNum = 500;
    int neighborhoodSize = 7;
    int migrationInterval = 1;
    int H = 299;

    if (argc == 5) {
        generationNum = std::stoi(argv[1]);
        neighborhoodSize = std::stoi(argv[2]);
        migrationInterval = std::stoi(argv[3]);
        H = std::stoi(argv[4]);
    }

    int totalPopulationSize = H + 1;

    auto problem = std::make_shared<ZDT1>();

    auto crossover = std::make_shared<SimulatedBinaryCrossover>(0.9, problem->VariableBounds());
    auto decomposition = std::make_shared<Tchebycheff>();
    auto mutation =
        std::make_shared<PolynomialMutation>(1.0 / problem->DecisionVariablesNum(), 20.0, problem->VariableBounds());
    auto sampling = std::make_shared<RealRandomSampling>(problem->VariableBounds());
    auto repair = std::make_shared<SamplingRepair<double>>(sampling);
    auto selection = std::make_shared<RandomSelection>();

    MpMoead<double> moead(generationNum, neighborhoodSize, H, migrationInterval, crossover, decomposition, mutation, problem,
                          repair, sampling, selection);

    double start = MPI_Wtime();

    moead.Run();

    double end = MPI_Wtime();
    double executionTime = end - start;
    double maxTime;
    MPI_Reduce(&executionTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "Maximum execution time across all processes: " << maxTime << " seconds" << std::endl;
    }

    std::filesystem::path objectiveFilePath = "out/data/mp_moead/objective/objective-" + std::to_string(rank) + ".txt";
    std::ofstream objectiveFile(objectiveFilePath);
    for (const auto& objectives : moead.GetObjectivesList()) {
        for (size_t j = 0; j < objectives.size(); j++) {
            objectiveFile << objectives[j] << " ";
        }
        objectiveFile << std::endl;
    }

    MPI_Finalize();

    return 0;
}