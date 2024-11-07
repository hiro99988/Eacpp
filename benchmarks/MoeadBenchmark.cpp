#include <mpi.h>

#include <chrono>
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

using namespace Eacpp;

std::ifstream OpenInputFile(const std::filesystem::path& filePath) {
    std::ifstream fileStream(filePath);
    if (!fileStream) {
        std::cerr << "Error: Could not open " << filePath << std::endl;
        std::exit(1);
    }
    return fileStream;
}

std::ofstream OpenOutputFile(const std::filesystem::path& filePath) {
    std::ofstream fileStream(filePath);
    if (!fileStream) {
        std::cerr << "Error: Could not create or open " << filePath << std::endl;
        std::exit(1);
    }
    return fileStream;
}

void CreateDirectories(const std::filesystem::path& dirPath) {
    try {
        std::filesystem::create_directories(dirPath);
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error: Could not create directories " << dirPath << std::endl;
        std::exit(1);
    }
}

std::string GetTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto nowTime = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&nowTime), "%y%m%d-%H%M%S");
    return ss.str();
}

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
    constexpr const char* ProblemsFilePath = "data/inputs/benchmarks/Problems.json";

    auto parameterFile = OpenInputFile(ParameterFilePath);
    auto problemsFile = OpenInputFile(ProblemsFilePath);

    nlohmann::json parameter = nlohmann::json::parse(parameterFile);
    nlohmann::json problems = nlohmann::json::parse(problemsFile);

    int trial = parameter["trial"];
    int generationNum = parameter["generationNum"];
    int neighborhoodSize = parameter["neighborhoodSize"];
    int divisionsNumOfWeightVector = parameter["divisionsNumOfWeightVector"];
    double crossoverRate = parameter["crossoverRate"];
    std::vector<std::string> problemNames = problems["problems"];

    const std::filesystem::path outputDirectoryPath = "out/data/Moead/" + GetTimestamp() + "/";
    CreateDirectories(outputDirectoryPath);

    for (auto&& problemName : problemNames) {
        const std::filesystem::path outputProblemDirectoryPath = outputDirectoryPath / problemName;
        const std::filesystem::path objectiveDirectoryPath = outputProblemDirectoryPath / "objective";
        const std::filesystem::path idealPointDirectoryPath = outputProblemDirectoryPath / "idealPoint";
        CreateDirectories(outputProblemDirectoryPath);
        CreateDirectories(objectiveDirectoryPath);
        CreateDirectories(idealPointDirectoryPath);
        const std::filesystem::path executionTimesFilePath = outputProblemDirectoryPath / "executionTimes.csv";
        std::ofstream executionTimesFile = OpenOutputFile(executionTimesFilePath);

        std::shared_ptr<IProblem<double>> problem = std::move(Reflection<IProblem<double>>::Create(problemName));
        auto crossover = std::make_shared<SimulatedBinaryCrossover>(crossoverRate, problem->VariableBounds());
        auto decomposition = std::make_shared<Tchebycheff>();
        auto mutation = std::make_shared<PolynomialMutation>(1.0 / problem->DecisionVariablesNum(), problem->VariableBounds());
        auto sampling = std::make_shared<RealRandomSampling>(problem->VariableBounds());
        auto repair = std::make_shared<RealRandomRepair>(problem);
        auto selection = std::make_shared<RandomSelection>();

        std::cout << "Problem: " << problemName << std::endl;

        for (int i = 0; i < trial; i++) {
            Moead<double> moead(generationNum, neighborhoodSize, divisionsNumOfWeightVector, crossover, decomposition, mutation,
                                problem, repair, sampling, selection);

            int generation = 0;
            double totalExecutionTime = 0.0;
            std::vector<std::pair<int, Eigen::ArrayXd>> transitionOfIdealPoint;

            auto start = std::chrono::high_resolution_clock::now();

            moead.Initialize();

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            totalExecutionTime += elapsed.count();

            AddIdealPoint(generation, decomposition->IdealPoint(), transitionOfIdealPoint);

            while (!moead.IsEnd()) {
                start = std::chrono::high_resolution_clock::now();

                moead.Update();

                end = std::chrono::high_resolution_clock::now();
                elapsed = end - start;
                totalExecutionTime += elapsed.count();

                generation++;
                AddIdealPoint(generation, decomposition->IdealPoint(), transitionOfIdealPoint);
            }

            std::cout << "Trial " << i + 1 << " Total execution time: " << totalExecutionTime << " seconds" << std::endl;

            std::filesystem::path objectiveFilePath = objectiveDirectoryPath / (std::to_string(i + 1) + ".csv");
            std::ofstream objectiveFile = OpenOutputFile(objectiveFilePath);
            for (const auto& objectives : moead.GetObjectivesList()) {
                for (size_t j = 0; j < objectives.size(); j++) {
                    if (j == objectives.size() - 1) {
                        objectiveFile << objectives[j];
                    } else {
                        objectiveFile << objectives[j] << ",";
                    }
                }
                objectiveFile << std::endl;
            }

            std::filesystem::path idealPointFilePath = idealPointDirectoryPath / (std::to_string(i + 1) + ".csv");
            std::ofstream idealPointFile = OpenOutputFile(idealPointFilePath);
            for (const auto& idealPoint : transitionOfIdealPoint) {
                idealPointFile << idealPoint.first << ",";
                for (size_t j = 0; j < idealPoint.second.size(); j++) {
                    if (j == idealPoint.second.size() - 1) {
                        idealPointFile << idealPoint.second(j);
                    } else {
                        idealPointFile << idealPoint.second(j) << ",";
                    }
                }
                idealPointFile << std::endl;
            }

            executionTimesFile << i + 1 << "," << totalExecutionTime << std::endl;
        }
    }

    return 0;
}