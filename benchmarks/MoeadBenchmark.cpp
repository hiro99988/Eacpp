#include <mpi.h>

#include <eigen3/Eigen/Core>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "Algorithms/Moead.h"
#include "Crossovers/SimulatedBinaryCrossover.h"
#include "Decompositions/Tchebycheff.h"
#include "Mutations/PolynomialMutation.h"
#include "Problems/Problems.h"
#include "Reflections/Reflection.h"
#include "Repairs/RealRandomRepair.h"
#include "Samplings/RealRandomSampling.h"
#include "Selections/RandomSelection.h"
#include "Stopwatches/Stopwatch.hpp"
#include "Utils/FileUtils.h"
#include "Utils/Utils.h"

using namespace Eacpp;

void AddIdealPoint(int gen, const Eigen::ArrayXd& add, std::vector<std::pair<int, Eigen::ArrayXd>>& vec) {
    if (vec.size() == 0) {
        vec.push_back(std::make_pair(gen, add));
        return;
    }

    if ((add != vec.back().second).any()) {
        vec.push_back(std::make_pair(gen, add));
    }
}

int main(int argc, char** argv) {
    constexpr const char* ParameterFilePath = "data/inputs/benchmarks/parameter.json";
    constexpr static std::array<const char*, 2> ExecutionTimesHeader = {"trial", "time(s)"};

    auto parameterFile = OpenInputFile(ParameterFilePath);

    nlohmann::json parameter = nlohmann::json::parse(parameterFile);

    int trial = parameter["trial"];
    int generationNum = parameter["generationNum"];
    int neighborhoodSize = parameter["neighborhoodSize"];
    int divisionsNumOfWeightVector = parameter["divisionsNumOfWeightVector"];
    double crossoverRate = parameter["crossoverRate"];
    std::vector<std::string> problemNames = parameter["problems"];

    const std::filesystem::path outputDirectoryPath = "out/data/Moead/" + GetTimestamp() + "/";
    std::filesystem::create_directories(outputDirectoryPath);

    std::filesystem::copy(ParameterFilePath, outputDirectoryPath / "parameter.json",
                          std::filesystem::copy_options::overwrite_existing);

    Stopwatch stopwatch;
    for (auto&& problemName : problemNames) {
        const std::filesystem::path outputProblemDirectoryPath = outputDirectoryPath / problemName;
        const std::filesystem::path objectiveDirectoryPath = outputProblemDirectoryPath / "objective";
        const std::filesystem::path idealPointDirectoryPath = outputProblemDirectoryPath / "idealPoint";
        std::filesystem::create_directories(outputProblemDirectoryPath);
        std::filesystem::create_directories(objectiveDirectoryPath);
        std::filesystem::create_directories(idealPointDirectoryPath);
        const std::filesystem::path executionTimesFilePath = outputProblemDirectoryPath / "executionTimes.csv";
        std::ofstream executionTimesFile = OpenOutputFile(executionTimesFilePath);
        WriteCsvLine(executionTimesFile, ExecutionTimesHeader);

        std::shared_ptr<IProblem<double>> problem = std::move(Reflection<IProblem<double>>::Create(problemName));
        auto crossover = std::make_shared<SimulatedBinaryCrossover>(crossoverRate, problem->VariableBounds());
        auto decomposition = std::make_shared<Tchebycheff>();
        auto mutation = std::make_shared<PolynomialMutation>(1.0 / problem->DecisionVariablesNum(), problem->VariableBounds());
        auto sampling = std::make_shared<RealRandomSampling>(problem->VariableBounds());
        auto repair = std::make_shared<RealRandomRepair>(problem);
        auto selection = std::make_shared<RandomSelection>();

        std::vector<std::string> objectiveHeader;
        for (int i = 0; i < problem->ObjectivesNum(); i++) {
            objectiveHeader.push_back("objective" + std::to_string(i + 1));
        }
        std::vector<std::string> idealPointHeader = {"generation"};
        for (int i = 0; i < problem->ObjectivesNum(); i++) {
            idealPointHeader.push_back("objective" + std::to_string(i + 1));
        }

        std::cout << "Problem: " << problemName << std::endl;

        for (int i = 0; i < trial; i++) {
            Moead<double> moead(generationNum, neighborhoodSize, divisionsNumOfWeightVector, crossover, decomposition, mutation,
                                problem, repair, sampling, selection);

            std::vector<std::pair<int, Eigen::ArrayXd>> transitionOfIdealPoint;

            stopwatch.Restart();

            moead.Initialize();

            stopwatch.Stop();
            AddIdealPoint(moead.CurrentGeneration(), decomposition->IdealPoint(), transitionOfIdealPoint);

            while (!moead.IsEnd()) {
                stopwatch.Start();

                moead.Update();

                stopwatch.Stop();
                AddIdealPoint(moead.CurrentGeneration(), decomposition->IdealPoint(), transitionOfIdealPoint);
            }

            std::cout << "Trial " << i + 1 << " Total execution time: " << stopwatch.Elapsed() << " seconds" << std::endl;

            std::filesystem::path objectiveFilePath = objectiveDirectoryPath / ("trial_" + std::to_string(i + 1) + ".csv");
            std::ofstream objectiveFile = OpenOutputFile(objectiveFilePath);
            WriteCsv(objectiveFile, moead.GetObjectivesList(), objectiveHeader);

            std::filesystem::path idealPointFilePath = idealPointDirectoryPath / ("trial_" + std::to_string(i + 1) + ".csv");
            std::ofstream idealPointFile = OpenOutputFile(idealPointFilePath);
            WriteCsv(idealPointFile, transitionOfIdealPoint, idealPointHeader);

            executionTimesFile << i + 1 << "," << stopwatch.Elapsed() << std::endl;
        }
    }

    return 0;
}