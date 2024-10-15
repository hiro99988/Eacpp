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

#include "Algorithms/MpMoead.h"
#include "Crossovers/SimulatedBinaryCrossover.h"
#include "Decompositions/Tchebycheff.h"
#include "Mutations/PolynomialMutation.h"
#include "Problems/Problems.h"
#include "Reflections/Reflection.h"
#include "Repairs/SamplingRepair.h"
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

int main(int argc, char** argv) {
    constexpr const char* ParameterFilePath = "data/inputs/benchmarks/MpMoead.json";
    constexpr const char* ProblemsFilePath = "data/inputs/benchmarks/Problems.json";

    int rank;
    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto parameterFile = OpenInputFile(ParameterFilePath);
    auto problemsFile = OpenInputFile(ProblemsFilePath);

    nlohmann::json parameter = nlohmann::json::parse(parameterFile);
    nlohmann::json problems = nlohmann::json::parse(problemsFile);

    int trial = parameter["trial"];
    int generationNum = parameter["generationNum"];
    int neighborhoodSize = parameter["neighborhoodSize"];
    int divisionsNumOfWeightVector = parameter["divisionsNumOfWeightVector"];
    int migrationInterval = parameter["migrationInterval"];
    double crossoverRate = parameter["crossoverRate"];
    std::vector<std::string> problemNames = problems["problems"];

    const std::filesystem::path outputDirectoryPath = "out/data/MpMoead/" + GetTimestamp() + "/";
    if (rank == 0) {
        CreateDirectories(outputDirectoryPath);
    }

    for (auto&& problemName : problemNames) {
        const std::filesystem::path outputProblemDirectoryPath = outputDirectoryPath / problemName;
        const std::filesystem::path objectiveDirectoryPath = outputProblemDirectoryPath / "objective";
        const std::filesystem::path idealPointDirectoryPath = outputProblemDirectoryPath / "idealPoint";
        if (rank == 0) {
            CreateDirectories(outputProblemDirectoryPath);
            CreateDirectories(objectiveDirectoryPath);
            CreateDirectories(idealPointDirectoryPath);
        }
        const std::filesystem::path executionTimesFilePath = outputProblemDirectoryPath / "executionTimes.csv";
        std::ofstream executionTimesFile;
        if (rank == 0) {
            executionTimesFile = OpenOutputFile(executionTimesFilePath);
        }

        std::shared_ptr<IProblem<double>> problem = std::move(Reflection<IProblem<double>>::Create(problemName));
        auto crossover = std::make_shared<SimulatedBinaryCrossover>(crossoverRate);
        auto decomposition = std::make_shared<Tchebycheff>();
        auto mutation = std::make_shared<PolynomialMutation>(1.0 / problem->DecisionVariablesNum(), problem->VariableBounds());
        auto sampling = std::make_shared<RealRandomSampling>(problem->VariableBounds());
        auto repair = std::make_shared<SamplingRepair<double>>(sampling);
        auto selection = std::make_shared<RandomSelection>();

        MpMoead<double> moead(generationNum, neighborhoodSize, divisionsNumOfWeightVector, migrationInterval, crossover,
                              decomposition, mutation, problem, repair, sampling, selection);
        if (rank == 0) {
            std::cout << "Problem: " << problemName << std::endl;
        }

        for (int i = 0; i < trial; i++) {
            double totalExecutionTime = 0.0;
            std::vector<Eigen::ArrayXd> transitionOfIdealPoint;

            MPI_Barrier(MPI_COMM_WORLD);
            double start = MPI_Wtime();

            moead.Initialize();

            double end = MPI_Wtime();
            totalExecutionTime += end - start;

            transitionOfIdealPoint.push_back(decomposition->IdealPoint());

            while (!moead.IsEnd()) {
                start = MPI_Wtime();

                moead.Update();

                end = MPI_Wtime();
                totalExecutionTime += end - start;

                transitionOfIdealPoint.push_back(decomposition->IdealPoint());
            }

            double maxTime;
            MPI_Reduce(&totalExecutionTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                std::cout << "Trial " << i + 1 << " Total execution time: " << totalExecutionTime << " seconds" << std::endl;
            }

            std::filesystem::path currentObjectiveDirectoryPath = objectiveDirectoryPath / (std::to_string(i + 1));
            std::filesystem::path currentIdealPointDirectoryPath = idealPointDirectoryPath / (std::to_string(i + 1));
            if (rank == 0) {
                CreateDirectories(currentObjectiveDirectoryPath);
                CreateDirectories(currentIdealPointDirectoryPath);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            std::filesystem::path objectiveFilePath = currentObjectiveDirectoryPath / (std::to_string(rank) + ".csv");
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

            std::filesystem::path idealPointFilePath = currentIdealPointDirectoryPath / (std::to_string(rank) + ".csv");
            std::ofstream idealPointFile = OpenOutputFile(idealPointFilePath);
            for (const auto& idealPoint : transitionOfIdealPoint) {
                for (size_t j = 0; j < idealPoint.size(); j++) {
                    if (j == idealPoint.size() - 1) {
                        idealPointFile << idealPoint[j];
                    } else {
                        idealPointFile << idealPoint[j] << ",";
                    }
                }
                idealPointFile << std::endl;
            }

            executionTimesFile << i + 1 << "," << totalExecutionTime << std::endl;
        }
    }

    MPI_Finalize();

    return 0;
}