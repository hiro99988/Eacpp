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
#include "Algorithms/MpMoeadIdealTopology.h"
#include "Algorithms/NewMoead.hpp"
#include "Algorithms/NtMoead.h"
#include "Algorithms/OneNtMoead.hpp"
#include "Crossovers/SimulatedBinaryCrossover.h"
#include "Decompositions/Tchebycheff.h"
#include "Indicators.hpp"
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
    constexpr static const char* DefaultParameterFilePath =
        "data/inputs/benchmarks/parameter.json";
    constexpr static std::array<const char*, 5> MoeadNames = {
        "MpMoead", "NtMoead", "NewMoead", "MpMoeadIdealTopology", "OneNtMoead"};
    constexpr static std::array<const char*, 2> ExecutionTimesHeader = {
        "trial", "time(s)"};
    constexpr static std::array<const char*, 2> IgdHeader = {"generation",
                                                             "igd"};

    int rank;
    int parallelSize;
    std::string moeadName;
    std::filesystem::path parameterFilePath;
    std::vector<std::pair<int, Eigen::ArrayXd>> transitionOfIdealPoint;
    std::vector<std::vector<Eigen::ArrayXd>> localObjectivesListHistory;

    ParallelMoeadBenchmark(const std::string& moeadName)
        : parameterFilePath(DefaultParameterFilePath) {
        InitializeMoeadName(moeadName);
    }

    ParallelMoeadBenchmark(const std::string& moeadName,
                           const std::filesystem::path& parameterFilePath)
        : parameterFilePath(parameterFilePath) {
        InitializeMoeadName(moeadName);
    }

    void InitializeMoeadName(const std::string& moeadName) {
        if (std::ranges::find(MoeadNames, moeadName) == MoeadNames.end()) {
            throw std::invalid_argument("Invalid moead name \"" + moeadName +
                                        "\"");
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

    nlohmann::json_abi_v3_11_3::json ReadParameters(
        std::ifstream& file, int& outTrial, int& outGenerationNum,
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

    void Warmup() {
        auto problem = std::make_shared<ZDT1>();
        auto crossover = std::make_shared<SimulatedBinaryCrossover>(
            0.9, problem->VariableBounds());
        auto decomposition = std::make_shared<Tchebycheff>();
        auto mutation = std::make_shared<PolynomialMutation>(
            1.0 / problem->DecisionVariablesNum(), problem->VariableBounds());
        auto sampling =
            std::make_shared<RealRandomSampling>(problem->VariableBounds());
        auto repair = std::make_shared<RealRandomRepair>(problem);
        auto selection = std::make_shared<RandomSelection>();

        auto moead = MpMoead<double>(10000, 21, 299, 1, crossover,
                                     decomposition, mutation, problem, repair,
                                     sampling, selection, true);

        moead.Run();

        MPI_Barrier(MPI_COMM_WORLD);
        ReleaseIsend(parallelSize, MPI_DOUBLE);
    }

    void GatherObjectivesListHistory(
        std::vector<std::vector<std::vector<double>>>& outObjectivesListHistory,
        std::vector<std::pair<int, std::vector<double>>>& finalObjectivesList) {
        if (rank == 0) {
            outObjectivesListHistory.reserve(localObjectivesListHistory.size());
        }

        for (int i = 0; i < localObjectivesListHistory.size(); i++) {
            const auto& localObjectivesList = localObjectivesListHistory[i];

            std::vector<double> sendBuffer;
            int dataSize = localObjectivesList[0].size();
            sendBuffer.reserve(dataSize * localObjectivesList.size());
            for (const auto& objectives : localObjectivesList) {
                sendBuffer.insert(sendBuffer.end(), objectives.begin(),
                                  objectives.end());
            }

            std::vector<double> receiveBuffer;
            std::vector<int> sizes;
            Gatherv(sendBuffer, rank, parallelSize, receiveBuffer, sizes);

            if (rank == 0) {
                std::vector<std::vector<double>> objectivesList;
                for (int j = 0, count = 0; j < parallelSize;
                     count += sizes[j], j++) {
                    for (int k = 0; k < sizes[j]; k += dataSize) {
                        std::vector<double> objectives(
                            receiveBuffer.begin() + count + k,
                            receiveBuffer.begin() + count + k + dataSize);

                        if (i == localObjectivesListHistory.size() - 1) {
                            objectivesList.push_back(objectives);
                            finalObjectivesList.push_back(
                                std::make_pair(j, std::move(objectives)));
                        } else {
                            objectivesList.push_back(std::move(objectives));
                        }
                    }
                }

                outObjectivesListHistory.push_back(std::move(objectivesList));
            }
        }
    }

    std::vector<std::pair<std::array<int, 2>, std::vector<double>>>
    GatherTransitionOfIdealPoint() {
        std::vector<double> sendBuffer;
        int dataSize = transitionOfIdealPoint[0].second.size() + 1;
        sendBuffer.reserve(dataSize * transitionOfIdealPoint.size());
        for (const auto& [gen, idealPoint] : transitionOfIdealPoint) {
            sendBuffer.push_back(gen);
            sendBuffer.insert(sendBuffer.end(), idealPoint.begin(),
                              idealPoint.end());
        }

        std::vector<double> receiveBuffer;
        std::vector<int> sizes;
        Gatherv(sendBuffer, rank, parallelSize, receiveBuffer, sizes);

        std::vector<std::pair<std::array<int, 2>, std::vector<double>>>
            transitionOfIdealPointList;
        if (rank == 0) {
            for (int i = 0, count = 0; i < parallelSize;
                 count += sizes[i], i++) {
                for (int j = 0; j < sizes[i]; j += dataSize) {
                    std::array<int, 2> keys = {
                        i, static_cast<int>(receiveBuffer[count + j])};
                    std::vector<double> objectives(
                        receiveBuffer.begin() + count + j + 1,
                        receiveBuffer.begin() + count + j + dataSize);
                    transitionOfIdealPointList.push_back(
                        std::make_pair(std::move(keys), std::move(objectives)));
                }
            }
        }

        return transitionOfIdealPointList;
    }

    void Run() {
        InitializeMpi();

        // パラメータ読み込み
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
        auto parameter = ReadParameters(
            parameterFile, trial, generationNum, neighborhoodSize,
            divisionsNumOfWeightVector, migrationInterval, crossoverRate,
            idealPointMigration, problemNames, adjacencyListFileName);

        // 出力ディレクトリの作成
        const std::filesystem::path outputDirectoryPath =
            "out/data/" + moeadName + "/" + GetTimestamp() + "/";
        RANK0(std::filesystem::create_directories(outputDirectoryPath);)

        // パラメータファイルのコピー
        if (rank == 0) {
            parameter["parallelSize"] = parallelSize;
            std::string parameterString = parameter.dump(4);
            std::ofstream parameterOutputFile(outputDirectoryPath /
                                              "parameter.json");
            parameterOutputFile << parameterString;
        }

        RANK0(std::cout << "Start warmup" << std::endl)
        Warmup();
        RANK0(std::cout << "End warmup" << std::endl)

        MpiStopwatch stopwatch;

        for (auto&& problemName : problemNames) {
            // 各種ディレクトリの作成
            const std::filesystem::path outputProblemDirectoryPath =
                outputDirectoryPath / problemName;
            const std::filesystem::path objectiveDirectoryPath =
                outputProblemDirectoryPath / "objective";
            const std::filesystem::path idealPointDirectoryPath =
                outputProblemDirectoryPath / "idealPoint";
            const std::filesystem::path igdDirectoryPath =
                outputProblemDirectoryPath / "igd";
            RANK0(
                std::filesystem::create_directories(outputProblemDirectoryPath);
                std::filesystem::create_directories(objectiveDirectoryPath);
                std::filesystem::create_directories(idealPointDirectoryPath);
                std::filesystem::create_directories(igdDirectoryPath);)

            // 実行時間ファイルの作成
            const std::filesystem::path executionTimesFilePath =
                outputProblemDirectoryPath / "executionTimes.csv";
            std::ofstream executionTimesFile;
            if (rank == 0) {
                executionTimesFile = OpenOutputFile(executionTimesFilePath);
                SetSignificantDigits(executionTimesFile, 9);
                WriteCsvLine(executionTimesFile, ExecutionTimesHeader);
            }

            // moeadの構成クラスの作成
            std::shared_ptr<IProblem<double>> problem =
                std::move(Reflection<IProblem<double>>::Create(problemName));
            auto crossover = std::make_shared<SimulatedBinaryCrossover>(
                crossoverRate, problem->VariableBounds());
            auto decomposition = std::make_shared<Tchebycheff>();
            auto mutation = std::make_shared<PolynomialMutation>(
                1.0 / problem->DecisionVariablesNum(),
                problem->VariableBounds());
            auto sampling =
                std::make_shared<RealRandomSampling>(problem->VariableBounds());
            auto repair = std::make_shared<RealRandomRepair>(problem);
            auto selection = std::make_shared<RandomSelection>();

            // ヘッダの作成
            std::vector<std::string> objectiveHeader = {"rank"};
            for (int i = 0; i < problem->ObjectivesNum(); i++) {
                objectiveHeader.push_back("objective" + std::to_string(i + 1));
            }
            std::vector<std::string> idealPointHeader = {"rank", "generation"};
            for (int i = 0; i < problem->ObjectivesNum(); i++) {
                idealPointHeader.push_back("objective" + std::to_string(i + 1));
            }

            // インディケータの作成
            std::ifstream paretoFrontFile;
            std::vector<std::vector<double>> paretoFront;
            if (rank == 0) {
                paretoFrontFile = OpenInputFile(
                    "data/ground_truth/pareto_fronts/" + problemName + ".csv");
                paretoFront = ReadCsv<double>(paretoFrontFile, false);
            }
            IGD indicator(paretoFront);

            RANK0(std::cout << "Problem: " << problemName << std::endl)

            for (int i = 0; i < trial; i++) {
                transitionOfIdealPoint.clear();
                localObjectivesListHistory.clear();

                std::unique_ptr<IMoead<double>> moead;
                if (moeadName == MoeadNames[0]) {
                    moead = std::make_unique<MpMoead<double>>(
                        generationNum, neighborhoodSize,
                        divisionsNumOfWeightVector, migrationInterval,
                        crossover, decomposition, mutation, problem, repair,
                        sampling, selection, idealPointMigration);
                } else if (moeadName == MoeadNames[1]) {
                    moead = std::make_unique<NtMoead<double>>(
                        generationNum, neighborhoodSize,
                        divisionsNumOfWeightVector, migrationInterval,
                        adjacencyListFileName, crossover, decomposition,
                        mutation, problem, repair, sampling, selection,
                        idealPointMigration);
                } else if (moeadName == MoeadNames[2]) {
                    moead = std::make_unique<NewMoead<double>>(
                        generationNum, neighborhoodSize,
                        divisionsNumOfWeightVector, migrationInterval,
                        "data/graph/new/neighboringMigration.csv",
                        "data/graph/new/idealPointMigration.csv", crossover,
                        decomposition, mutation, problem, repair, sampling,
                        selection, idealPointMigration);
                } else if (moeadName == MoeadNames[3]) {
                    moead = std::make_unique<MpMoeadIdealTopology<double>>(
                        generationNum, neighborhoodSize,
                        divisionsNumOfWeightVector, migrationInterval,
                        "data/graph/2_13_29_10_3_100_100/adjacencyList.csv",
                        crossover, decomposition, mutation, problem, repair,
                        sampling, selection, idealPointMigration);
                } else if (moeadName == MoeadNames[4]) {
                    moead = std::make_unique<OneNtMoead<double>>(
                        generationNum, neighborhoodSize,
                        divisionsNumOfWeightVector, migrationInterval,
                        adjacencyListFileName, crossover, decomposition,
                        mutation, problem, repair, sampling, selection,
                        idealPointMigration);
                } else {
                    throw std::invalid_argument("Invalid moead name");
                }

                MPI_Barrier(MPI_COMM_WORLD);
                stopwatch.Restart();

                moead->Initialize();

                stopwatch.Stop();

                AddIdealPoint(moead->CurrentGeneration(),
                              decomposition->IdealPoint());
                localObjectivesListHistory.push_back(
                    moead->GetObjectivesList());

                while (!moead->IsEnd()) {
                    stopwatch.Start();

                    moead->Update();

                    stopwatch.Stop();

                    AddIdealPoint(moead->CurrentGeneration(),
                                  decomposition->IdealPoint());
                    localObjectivesListHistory.push_back(
                        moead->GetObjectivesList());
                }

                // 実行時間の出力
                double elapsed = stopwatch.Elapsed();
                double maxExecutionTime;
                MPI_Reduce(&elapsed, &maxExecutionTime, 1, MPI_DOUBLE, MPI_MAX,
                           0, MPI_COMM_WORLD);
                RANK0(std::cout << "Trial " << i + 1
                                << " Total execution time: " << maxExecutionTime
                                << " seconds" << std::endl;)
                RANK0(executionTimesFile << i + 1 << "," << maxExecutionTime
                                         << std::endl;)

                std::string fileName =
                    "trial_" + std::to_string(i + 1) + ".csv";

                // 理想点の出力
                auto transitionOfIdealPointList =
                    GatherTransitionOfIdealPoint();
                if (rank == 0) {
                    std::filesystem::path idealPointFilePath =
                        idealPointDirectoryPath / fileName;
                    std::ofstream idealPointFile =
                        OpenOutputFile(idealPointFilePath);
                    SetSignificantDigits(idealPointFile);
                    WriteCsv(idealPointFile, transitionOfIdealPointList,
                             idealPointHeader);
                }

                // 目的関数値の出力
                std::vector<std::vector<std::vector<double>>>
                    objectivesListHistory;
                std::vector<std::pair<int, std::vector<double>>>
                    finalObjectivesList;
                GatherObjectivesListHistory(objectivesListHistory,
                                            finalObjectivesList);
                if (rank == 0) {
                    std::filesystem::path objectiveFilePath =
                        objectiveDirectoryPath / fileName;
                    std::ofstream objectiveFile =
                        OpenOutputFile(objectiveFilePath);
                    SetSignificantDigits(objectiveFile);
                    WriteCsv(objectiveFile, finalObjectivesList,
                             objectiveHeader);
                }

                // IGDの出力
                if (rank == 0) {
                    std::filesystem::path igdFilePath =
                        igdDirectoryPath / fileName;
                    std::ofstream igdFile = OpenOutputFile(igdFilePath);
                    SetSignificantDigits(igdFile);
                    std::vector<std::pair<int, double>> igd;
                    for (int j = 0; j < objectivesListHistory.size(); j++) {
                        igd.push_back(std::make_pair(
                            j, indicator.Calculate(objectivesListHistory[j])));
                    }
                    WriteCsv(igdFile, igd, IgdHeader);
                }

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
        RANK0(std::cerr << "Usage: " << argv[0]
                        << " <moeadName> [parameterFilePath]" << std::endl;)
        MPI_Finalize();
        return 1;
    } else if (argc == 2) {
        moeadName = argv[1];
    } else if (argc == 3) {
        moeadName = argv[1];
        parameterFilePath = argv[2];
    }

    auto benchmark =
        parameterFilePath.empty()
            ? Eacpp::ParallelMoeadBenchmark(moeadName)
            : Eacpp::ParallelMoeadBenchmark(moeadName, parameterFilePath);
    benchmark.Run();

    MPI_Finalize();
    return 0;
}