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
#include "Indicators.hpp"
#include "Mutations/PolynomialMutation.h"
#include "Problems/Problems.h"
#include "Reflections/Reflection.h"
#include "Repairs/RealRandomRepair.h"
#include "Samplings/RealRandomSampling.h"
#include "Selections/RandomSelection.h"
#include "Utils/FileUtils.h"
#include "Utils/MpiUtils.h"

using namespace eacpp;

int main(int argc, char** argv) {
    int rank, parallelSize;
    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &parallelSize);

    int generationNum = 500;
    int H = 299;
    int neighborhoodSize = 7;
    int migrationInterval = 1;
    bool isAsync = true;

    if (argc == 2) {
        generationNum = std::stoi(argv[1]);
    } else if (argc == 3) {
        generationNum = std::stoi(argv[1]);
        H = std::stoi(argv[2]);
    } else if (argc == 4) {
        generationNum = std::stoi(argv[1]);
        H = std::stoi(argv[2]);
        neighborhoodSize = std::stoi(argv[3]);
    } else if (argc == 5) {
        generationNum = std::stoi(argv[1]);
        H = std::stoi(argv[2]);
        neighborhoodSize = std::stoi(argv[3]);
        migrationInterval = std::stoi(argv[4]);
    } else if (argc == 6) {
        generationNum = std::stoi(argv[1]);
        H = std::stoi(argv[2]);
        neighborhoodSize = std::stoi(argv[3]);
        migrationInterval = std::stoi(argv[4]);
        isAsync = static_cast<bool>(std::stoi(argv[5]));
    }

    std::shared_ptr<IProblem<double>> problem = CreateProblem("zdt1", 30, 2);

    auto crossover = std::make_shared<SimulatedBinaryCrossover>(
        0.9, problem->VariableBounds());
    auto decomposition = std::make_shared<Tchebycheff>();
    auto mutation = std::make_shared<PolynomialMutation>(
        1.0 / problem->DecisionVariablesNum(), 20.0, problem->VariableBounds());
    auto sampling =
        std::make_shared<RealRandomSampling>(problem->VariableBounds());
    auto repair = std::make_shared<RealRandomRepair>(problem);
    auto selection = std::make_shared<RandomSelection>();

    MpMoead<double> moead(generationNum, neighborhoodSize, H, migrationInterval,
                          crossover, decomposition, mutation, problem, repair,
                          sampling, selection, isAsync);

    double start = MPI_Wtime();
    moead.Run();
    double end = MPI_Wtime();

    double elapsedTime = end - start;
    double maxTime;
    MPI_Reduce(&elapsedTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "Elapsed time: " << maxTime << " seconds" << std::endl;
    }

    std::vector<std::vector<double>> paretoFront;
    if (rank == 0) {
        std::filesystem::path paretoFrontFilePath =
            "data/ground_truth/pareto_fronts/ZDT1.csv";
        auto paretoFrontFile = OpenInputFile(paretoFrontFilePath);
        paretoFront = ReadCsv<double>(paretoFrontFile, true, true);
    }
    IGDPlus indicator(paretoFront);
    auto objectivesList = moead.GetObjectivesList();
    std::vector<double> localObjectives;
    localObjectives.reserve(objectivesList.size() * problem->ObjectivesNum());
    for (const auto& obj : objectivesList) {
        localObjectives.insert(localObjectives.end(), obj.data(),
                               obj.data() + obj.size());
    }
    std::vector<double> allObjectives;
    std::vector<int> sizes;
    Gatherv(localObjectives, rank, parallelSize, allObjectives, sizes);
    if (rank == 0) {
        std::vector<Eigen::ArrayXd> gatheredObjectivesList;
        for (size_t i = 0; i < allObjectives.size();
             i += problem->ObjectivesNum()) {
            Eigen::ArrayXd obj(problem->ObjectivesNum());
            for (int j = 0; j < problem->ObjectivesNum(); ++j) {
                obj[j] = allObjectives[i + j];
            }
            gatheredObjectivesList.push_back(std::move(obj));
        }
        double igdPlusValue = indicator.Calculate(gatheredObjectivesList);
        std::cout << "IGD+: " << igdPlusValue << std::endl;
    }

    MPI_Finalize();

    return 0;
}