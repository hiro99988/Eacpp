#include <mpi.h>

#include <eigen3/Eigen/Core>
#include <filesystem>
#include <iostream>
#include <memory>

#include "Algorithms/Moead.h"
#include "Algorithms/MpMoead.h"
#include "Crossovers/SimulatedBinaryCrossover.h"
#include "Decompositions/Tchebycheff.h"
#include "Mutations/PolynomialMutation.h"
#include "Problems/Problems.h"
#include "Problems/ZDT1.h"
#include "Repairs/SamplingRepair.h"
#include "Samplings/UniformRandomSampling.h"
#include "Selections/RandomSelection.h"

using namespace Eacpp;

int main(int argc, char** argv) {
    constexpr char* inputFilePath = "data/inputs/benchmarks/Moead.txt";

    int trial;
    int generationNum;
    int neighborNum;
    int H;
    double crossoverRate;

    std::ifstream inputFile(inputFilePath);
    if (!inputFile) {
        std::cerr << "Error opening input file: " << inputFilePath << std::endl;
        return 1;
    }

    inputFile >> trial;
    inputFile >> generationNum;
    inputFile >> neighborNum;
    inputFile >> H;
    inputFile >> crossoverRate;

    const char* outputDirectoryPath = "out/data/moead/";
    const char* executionTimeFilePath = "out/data/moead/execution_time.txt";
    const char* objectiveDirectoryPath = "out/data/moead/objective";
    const char* idealPointDirectoryPath = "out/data/moead/ideal_point";

    auto problem = std::make_shared<ZDT1>();
    int decisionVariableNum = problem->decisionVariablesNum;
    int objectiveNum = problem->objectivesNum;
    std::vector<std::pair<double, double>> variableBound{problem->variableBound};
    auto crossover = std::make_shared<SimulatedBinaryCrossover>(crossoverRate, 20.0);
    auto decomposition = std::make_shared<Tchebycheff>(objectiveNum);
    auto mutation = std::make_shared<PolynomialMutation>(1.0 / decisionVariableNum, 20.0, variableBound);
    auto sampling = std::make_shared<UniformRandomSampling>(problem->variableBound.first, problem->variableBound.second);
    auto repair = std::make_shared<SamplingRepair<double>>(sampling);
    auto selection = std::make_shared<RandomSelection>();

    std::ofstream executionTimeFile(executionTimeFilePath);
    if (!executionTimeFile) {
        std::cerr << "Error opening execution time file: " << executionTimeFilePath << std::endl;
        return 1;
    }

    std::chrono::duration<double> elapsed;
    for (int i = 0; i < trial; ++i) {
        double totalExecutionTime = 0.0;
        std::vector<Eigen::ArrayXd> transitionOfIdealPoint;

        Moead<double> moead(generationNum, decisionVariableNum, objectiveNum, neighborNum, H, crossover, decomposition,
                            mutation, problem, repair, sampling, selection);

        auto start = std::chrono::high_resolution_clock::now();

        moead.Initialize();

        auto end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        totalExecutionTime += elapsed.count();

        transitionOfIdealPoint.push_back(decomposition->IdealPoint());

        for (int gen = 0; gen < generationNum; ++gen) {
            start = std::chrono::high_resolution_clock::now();

            moead.Update();

            end = std::chrono::high_resolution_clock::now();
            elapsed = end - start;
            totalExecutionTime += elapsed.count();

            transitionOfIdealPoint.push_back(decomposition->IdealPoint());
        }

        std::cout << "Trial " << trial << " Total execution time: " << totalExecutionTime << " seconds" << std::endl;
        executionTimeFile << i + 1 << "," << totalExecutionTime << std::endl;

        std::filesystem::create_directories("out/data/");
        std::filesystem::create_directories("out/data/moead");
        std::ofstream ofs("out/data/moead/result.txt");
        for (const auto& objectives : moead.GetObjectivesList()) {
            for (const auto& value : objectives) {
                ofs << value << " ";
            }
            ofs << std::endl;
        }
    }

    return 0;
}