#include <mpi.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <tuple>
#include <vector>

#include "Algorithms/IParallelMoead.hpp"
#include "Algorithms/MpMoeadIdealTopology.h"
#include "Crossovers/SimulatedBinaryCrossover.h"
#include "Decompositions/Tchebycheff.h"
#include "Mutations/PolynomialMutation.h"
#include "Problems/ZDT1.h"
#include "Reflections/Reflection.h"
#include "Repairs/RealRandomRepair.h"
#include "Samplings/RealRandomSampling.h"
#include "Selections/RandomSelection.h"

using namespace Eacpp;

int main(int argc, char** argv) {
    int rank;
    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::ifstream file("data/inputs/exampleParameter.json");
    nlohmann::json parameter = nlohmann::json::parse(file);

    int generationNum = parameter["generationNum"];
    int neighborhoodSize = parameter["neighborhoodSize"];
    int divisionsNumOfWeightVector = parameter["divisionsNumOfWeightVector"];
    int migrationInterval = parameter["migrationInterval"];
    std::string problemName = parameter["problem"];
    std::string adjacencyListFileName = parameter["adjacencyListFileName"];
    bool isAsync = parameter["isAsync"];

    std::shared_ptr<IProblem<double>> problem =
        Reflection<IProblem<double>>::Create(problemName);

    auto crossover = std::make_shared<SimulatedBinaryCrossover>(
        0.9, problem->VariableBounds());
    auto decomposition = std::make_shared<Tchebycheff>();
    auto mutation = std::make_shared<PolynomialMutation>(
        1.0 / problem->DecisionVariablesNum(), problem->VariableBounds());
    auto sampling =
        std::make_shared<RealRandomSampling>(problem->VariableBounds());
    auto repair = std::make_shared<RealRandomRepair>(problem);
    auto selection = std::make_shared<RandomSelection>();

    auto moead = MpMoeadIdealTopology<double>(
        generationNum, neighborhoodSize, divisionsNumOfWeightVector,
        migrationInterval, adjacencyListFileName, crossover, decomposition,
        mutation, problem, repair, sampling, selection, isAsync);

    double start = MPI_Wtime();
    moead.Run();
    double end = MPI_Wtime();
    double executionTime = end - start;
    double maxTime;
    MPI_Reduce(&executionTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "Maximum execution time across all processes: " << maxTime
                  << " seconds" << std::endl;
    }

    // std::filesystem::path objectiveFilePath =
    //     "out/data/tmp/objective/" + std::to_string(rank) + ".csv";
    // std::ofstream objectiveFile(objectiveFilePath);
    // for (const auto& objectives : moead.GetObjectivesList()) {
    //     for (int i = 0; i < objectives.size(); i++) {
    //         objectiveFile << objectives[i];
    //         if (i != objectives.size() - 1) {
    //             objectiveFile << ",";
    //         }
    //     }
    //     objectiveFile << std::endl;
    // }

    MPI_Finalize();

    return 0;
}