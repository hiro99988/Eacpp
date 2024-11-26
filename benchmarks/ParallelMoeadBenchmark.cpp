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
#include "Algorithms/NtMoead.h"
#include "Crossovers/SimulatedBinaryCrossover.h"
#include "Decompositions/Tchebycheff.h"
#include "Mutations/PolynomialMutation.h"
#include "Problems/Problems.h"
#include "Reflections/Reflection.h"
#include "Repairs/RealRandomRepair.h"
#include "Samplings/RealRandomSampling.h"
#include "Selections/RandomSelection.h"
#include "Stopwatches/MpiStopwatch.hpp"
#include "Utils/FileUtils.h"
#include "Utils/MpiUtils.h"
#include "Utils/Utils.h"

namespace Eacpp {

class ParallelMoeadBenchmark {
   public:
    constexpr static const char* DefaultParameterFilePath = "data/inputs/benchmarks/parameter.json";
    constexpr static std::array<const char*, 2> MoeadNames = {"MpMoead", "NtMoead"};
    constexpr static std::array<const char*, 2> ExecutionTimesHeader = {"trial", "time(s)"};

    int rank;
    int parallelSize;
    std::string moeadName;
    std::filesystem::path parameterFilePath;
    std::vector<std::pair<int, Eigen::ArrayXd>> transitionOfIdealPoint;

    ParallelMoeadBenchmark(const std::string& moeadName) : parameterFilePath(DefaultParameterFilePath) {
        InitializeMoeadName(moeadName);
    }
    ParallelMoeadBenchmark(const std::string& moeadName, const std::filesystem::path& parameterFilePath)
        : parameterFilePath(parameterFilePath) {
        InitializeMoeadName(moeadName);
    }

    void InitializeMoeadName(const std::string& moeadName) {
        if (std::ranges::find(MoeadNames, moeadName) == MoeadNames.end()) {
            throw std::invalid_argument("Invalid moead name \"" + moeadName + "\"");
        }

        this->moeadName = moeadName;
    }

    void InitializeMpi() {
        int initialized;
        MPI_Initialized(&initialized);
        if (!initialized) {
            MPI_Init(nullptr, nullptr);
        }
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &parallelSize);
    }

    nlohmann::json_abi_v3_11_3::json ReadParameters(std::ifstream& file, int& outTrial, int& outGenerationNum,
                                                    int& outNeighborhoodSize, int& outDivisionsNumOfWeightVector,
                                                    int& outMigrationInterval, double& outCrossoverRate,
                                                    bool& outIdealPointMigration, std::vector<std::string>& outProblemNames,
                                                    std::string& outAdjacencyListFileName) {
        nlohmann::json parameter = nlohmann::json::parse(file);

        outTrial = parameter["trial"];
        outGenerationNum = parameter["generationNum"];
        outNeighborhoodSize = parameter["neighborhoodSize"];
        outDivisionsNumOfWeightVector = parameter["divisionsNumOfWeightVector"];
        outMigrationInterval = parameter["migrationInterval"];
        outCrossoverRate = parameter["crossoverRate"];
        outIdealPointMigration = parameter["idealPointMigration"];
        outProblemNames = parameter["problems"];
        outAdjacencyListFileName = parameter["adjacencyListFileName"];

        return parameter;
    }

    void AddIdealPoint(int gen, const Eigen::ArrayXd& add) {
        if (transitionOfIdealPoint.size() == 0) {
            transitionOfIdealPoint.push_back(std::make_pair(gen, add));
            return;
        }

        if ((add != transitionOfIdealPoint.back().second).any()) {
            transitionOfIdealPoint.push_back(std::make_pair(gen, add));
        }
    }

    std::vector<std::pair<int, std::vector<double>>> GatherObjectivesList(
        const std::vector<Eigen::ArrayXd>& localObjectivesList) {
        std::vector<double> sendBuffer;
        int dataSize = localObjectivesList[0].size();
        sendBuffer.reserve(dataSize * localObjectivesList.size());
        for (const auto& objectives : localObjectivesList) {
            sendBuffer.insert(sendBuffer.end(), objectives.begin(), objectives.end());
        }

        std::vector<double> receiveBuffer;
        std::vector<int> sizes;
        Gatherv(sendBuffer, rank, parallelSize, receiveBuffer, sizes);

        std::vector<std::pair<int, std::vector<double>>> objectivesList;
        if (rank == 0) {
            for (int i = 0, count = 0; i < parallelSize; count += sizes[i], i++) {
                for (int j = 0; j < sizes[i]; j += dataSize) {
                    std::vector<double> objectives(receiveBuffer.begin() + count + j,
                                                   receiveBuffer.begin() + count + j + dataSize);
                    objectivesList.push_back(std::make_pair(i, std::move(objectives)));
                }
            }
        }

        return objectivesList;
    }

    std::vector<std::pair<std::array<int, 2>, std::vector<double>>> GatherTransitionOfIdealPoint() {
        std::vector<double> sendBuffer;
        int dataSize = transitionOfIdealPoint[0].second.size() + 1;
        sendBuffer.reserve(dataSize * transitionOfIdealPoint.size());
        for (const auto& [gen, idealPoint] : transitionOfIdealPoint) {
            sendBuffer.push_back(gen);
            sendBuffer.insert(sendBuffer.end(), idealPoint.begin(), idealPoint.end());
        }

        std::vector<double> receiveBuffer;
        std::vector<int> sizes;
        Gatherv(sendBuffer, rank, parallelSize, receiveBuffer, sizes);

        std::vector<std::pair<std::array<int, 2>, std::vector<double>>> transitionOfIdealPointList;
        if (rank == 0) {
            for (int i = 0, count = 0; i < parallelSize; count += sizes[i], i++) {
                for (int j = 0; j < sizes[i]; j += dataSize) {
                    std::array<int, 2> keys = {i, static_cast<int>(receiveBuffer[count + j])};
                    std::vector<double> objectives(receiveBuffer.begin() + count + j + 1,
                                                   receiveBuffer.begin() + count + j + dataSize);
                    transitionOfIdealPointList.push_back(std::make_pair(std::move(keys), std::move(objectives)));
                }
            }
        }

        return transitionOfIdealPointList;
    }

    void Run() {
        InitializeMpi();

        int trial;
        int generationNum;
        int neighborhoodSize;
        int divisionsNumOfWeightVector;
        int migrationInterval;
        double crossoverRate;
        bool idealPointMigration;
        std::vector<std::string> problemNames;
        std::string adjacencyListFileName;
        auto parameterFile = OpenInputFile(parameterFilePath);
        auto parameter =
            ReadParameters(parameterFile, trial, generationNum, neighborhoodSize, divisionsNumOfWeightVector, migrationInterval,
                           crossoverRate, idealPointMigration, problemNames, adjacencyListFileName);

        const std::filesystem::path outputDirectoryPath = "out/data/" + moeadName + "/" + GetTimestamp() + "/";
        RANK0(std::filesystem::create_directories(outputDirectoryPath);)

        if (rank == 0) {
            parameter["parallelSize"] = parallelSize;
            std::string parameterString = parameter.dump(4);
            std::ofstream parameterOutputFile(outputDirectoryPath / "parameter.json");
            parameterOutputFile << parameterString;
        }

        MpiStopwatch stopwatch;
        for (auto&& problemName : problemNames) {
            const std::filesystem::path outputProblemDirectoryPath = outputDirectoryPath / problemName;
            const std::filesystem::path objectiveDirectoryPath = outputProblemDirectoryPath / "objective";
            const std::filesystem::path idealPointDirectoryPath = outputProblemDirectoryPath / "idealPoint";
            RANK0(std::filesystem::create_directories(outputProblemDirectoryPath);
                  std::filesystem::create_directories(objectiveDirectoryPath);
                  std::filesystem::create_directories(idealPointDirectoryPath);)

            const std::filesystem::path executionTimesFilePath = outputProblemDirectoryPath / "executionTimes.csv";
            std::ofstream executionTimesFile;
            RANK0(executionTimesFile = OpenOutputFile(executionTimesFilePath);
                  WriteCsvLine(executionTimesFile, ExecutionTimesHeader);)

            std::shared_ptr<IProblem<double>> problem = std::move(Reflection<IProblem<double>>::Create(problemName));
            auto crossover = std::make_shared<SimulatedBinaryCrossover>(crossoverRate, problem->VariableBounds());
            auto decomposition = std::make_shared<Tchebycheff>();
            auto mutation =
                std::make_shared<PolynomialMutation>(1.0 / problem->DecisionVariablesNum(), problem->VariableBounds());
            auto sampling = std::make_shared<RealRandomSampling>(problem->VariableBounds());
            auto repair = std::make_shared<RealRandomRepair>(problem);
            auto selection = std::make_shared<RandomSelection>();

            std::vector<std::string> objectiveHeader = {"rank"};
            for (int i = 0; i < problem->ObjectivesNum(); i++) {
                objectiveHeader.push_back("objective" + std::to_string(i + 1));
            }
            std::vector<std::string> idealPointHeader = {"rank", "generation"};
            for (int i = 0; i < problem->ObjectivesNum(); i++) {
                idealPointHeader.push_back("objective" + std::to_string(i + 1));
            }

            RANK0(std::cout << "Problem: " << problemName << std::endl)

            for (int i = 0; i < trial; i++) {
                transitionOfIdealPoint.clear();
                std::unique_ptr<IMoead<double>> moead;
                if (moeadName == MoeadNames[0]) {
                    moead = std::make_unique<MpMoead<double>>(generationNum, neighborhoodSize, divisionsNumOfWeightVector,
                                                              migrationInterval, crossover, decomposition, mutation, problem,
                                                              repair, sampling, selection, idealPointMigration);
                } else if (moeadName == MoeadNames[1]) {
                    moead = std::make_unique<NtMoead<double>>(
                        generationNum, neighborhoodSize, divisionsNumOfWeightVector, migrationInterval, adjacencyListFileName,
                        crossover, decomposition, mutation, problem, repair, sampling, selection, idealPointMigration);
                } else {
                    throw std::invalid_argument("Invalid moead name");
                }

                MPI_Barrier(MPI_COMM_WORLD);
                stopwatch.Restart();

                moead->Initialize();

                stopwatch.Stop();
                AddIdealPoint(moead->CurrentGeneration(), decomposition->IdealPoint());

                while (!moead->IsEnd()) {
                    stopwatch.Start();

                    moead->Update();

                    stopwatch.Stop();
                    AddIdealPoint(moead->CurrentGeneration(), decomposition->IdealPoint());
                }

                double elapsed = stopwatch.Elapsed();
                double maxExecutionTime;
                MPI_Reduce(&elapsed, &maxExecutionTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                RANK0(std::cout << "Trial " << i + 1 << " Total execution time: " << maxExecutionTime << " seconds"
                                << std::endl;)
                RANK0(executionTimesFile << i + 1 << "," << maxExecutionTime << std::endl;)

                std::filesystem::path objectiveFilePath = objectiveDirectoryPath / ("trial_" + std::to_string(i + 1) + ".csv");
                std::ofstream objectiveFile = OpenOutputFile(objectiveFilePath);
                auto localObjectivesList = moead->GetObjectivesList();
                auto objectivesList = GatherObjectivesList(localObjectivesList);
                RANK0(WriteCsv(objectiveFile, objectivesList, objectiveHeader);)

                std::filesystem::path idealPointFilePath =
                    idealPointDirectoryPath / ("trial_" + std::to_string(i + 1) + ".csv");
                std::ofstream idealPointFile = OpenOutputFile(idealPointFilePath);
                auto transitionOfIdealPointList = GatherTransitionOfIdealPoint();
                RANK0(WriteCsv(idealPointFile, transitionOfIdealPointList, idealPointHeader);)

                MPI_Barrier(MPI_COMM_WORLD);
                ReleaseIsend(parallelSize, MPI_DOUBLE);
            }
        }
    }
};

}  // namespace Eacpp

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::string moeadName;
    std::string parameterFilePath;
    if (argc < 2 || argc > 4) {
        RANK0(std::cerr << "Usage: " << argv[0] << " <moeadName> [parameterFilePath]" << std::endl;)
        MPI_Finalize();
        return 1;
    } else if (argc == 2) {
        moeadName = argv[1];
    } else if (argc == 3) {
        moeadName = argv[1];
        parameterFilePath = argv[2];
    }

    auto benchmark = parameterFilePath.empty() ? Eacpp::ParallelMoeadBenchmark(moeadName)
                                               : Eacpp::ParallelMoeadBenchmark(moeadName, parameterFilePath);
    benchmark.Run();

    MPI_Finalize();
    return 0;
}