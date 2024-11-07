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

#include "Algorithms/NtMoead.h"
#include "Crossovers/SimulatedBinaryCrossover.h"
#include "Decompositions/Tchebycheff.h"
#include "Mutations/PolynomialMutation.h"
#include "Problems/Problems.h"
#include "Reflections/Reflection.h"
#include "Repairs/RealRandomRepair.h"
#include "Samplings/RealRandomSampling.h"
#include "Selections/RandomSelection.h"

#define RANK0(code)  \
    if (rank == 0) { \
        code;        \
    }

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

void ReleaseIsend(int parallelSize) {
    for (int i = 0; i < parallelSize; i++) {
        while (true) {
            int flag;
            MPI_Status status;
            MPI_Iprobe(i, 0, MPI_COMM_WORLD, &flag, &status);
            if (!flag) {
                break;
            }

            int dataSize;
            MPI_Get_count(&status, MPI_DOUBLE, &dataSize);
            std::vector<double> tempData(dataSize);
            MPI_Recv(tempData.data(), dataSize, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
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

    int rank;
    int parallelSize;
    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &parallelSize);

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

    const std::filesystem::path outputDirectoryPath = "out/data/NtMoead/" + GetTimestamp() + "/";
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
        auto crossover = std::make_shared<SimulatedBinaryCrossover>(crossoverRate, problem->VariableBounds());
        auto decomposition = std::make_shared<Tchebycheff>();
        auto mutation = std::make_shared<PolynomialMutation>(1.0 / problem->DecisionVariablesNum(), problem->VariableBounds());
        auto sampling = std::make_shared<RealRandomSampling>(problem->VariableBounds());
        auto repair = std::make_shared<RealRandomRepair>(problem);
        auto selection = std::make_shared<RandomSelection>();

        RANK0(std::cout << "Problem: " << problemName << std::endl;)

        for (int i = 0; i < trial; i++) {
            NtMoead<double> moead(generationNum, neighborhoodSize, divisionsNumOfWeightVector, migrationInterval, crossover,
                                  decomposition, mutation, problem, repair, sampling, selection);

            ReleaseIsend(parallelSize);

            int generation = 0;
            double totalExecutionTime = 0.0;
            std::vector<std::pair<int, Eigen::ArrayXd>> transitionOfIdealPoint;

            MPI_Barrier(MPI_COMM_WORLD);
            double start = MPI_Wtime();

            moead.Initialize();

            double end = MPI_Wtime();
            totalExecutionTime += end - start;

            AddIdealPoint(generation, decomposition->IdealPoint(), transitionOfIdealPoint);

            while (!moead.IsEnd()) {
                start = MPI_Wtime();

                moead.Update();

                end = MPI_Wtime();
                totalExecutionTime += end - start;

                generation++;
                AddIdealPoint(generation, decomposition->IdealPoint(), transitionOfIdealPoint);
            }

            double maxTime;
            MPI_Reduce(&totalExecutionTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            RANK0(std::cout << "Trial " << i + 1 << " Total execution time: " << totalExecutionTime << " seconds" << std::endl;)

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

    MPI_Finalize();

    return 0;
}