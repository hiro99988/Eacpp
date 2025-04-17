#include <mpi.h>

#include <Eigen/Core>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "Algorithms/HalfMpMoead.hpp"
#include "Algorithms/MpMoead.h"
#include "Algorithms/MpMoeadIdealTopology.h"
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
        "data/inputs/parameter.json";
    constexpr static std::array<const char*, 3> MoeadNames = {
        "MpMoead", "MpMoeadIt", "HalfMpMoead"};
    constexpr static std::array<const char*, 2> ElapsedTimeHeaders = {
        "trial", "time(s)"};
    constexpr static std::array<const char*, 3> IgdHeader = {
        "generation", "executionTime(s)", "igd"};
    constexpr static std::array<const char*, 6> DataTrafficsHeader = {
        "rank",         "generation",
        "sendTimes",    "totalSendDataTraffic",
        "receiveTimes", "totalReceiveDataTraffic"};

    int rank;
    int parallelSize;
    std::string moeadName;
    std::filesystem::path parameterFilePath;
    std::vector<std::pair<int, Eigen::ArrayXd>> transitionOfIdealPoint;
    std::vector<std::vector<Eigen::ArrayXd>> localObjectivesListHistory;
    std::vector<double> executionTimes;

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

    nlohmann::json ReadParameters(std::ifstream& file, int& outTrial,
                                  int& outGenerationNum,
                                  int& outNeighborhoodSize,
                                  int& outDivisionsNumOfWeightVector,
                                  int& outMigrationInterval,
                                  double& outCrossoverRate,
                                  int& outObjectivesNum, bool& outIsAsync,
                                  std::vector<int>& outDecisionVariablesNums,
                                  std::vector<std::string>& outProblemNames,
                                  std::string& outAdjacencyListFileName) {
        nlohmann::json parameter = nlohmann::json::parse(file);

        outTrial = parameter["trial"];
        outGenerationNum = parameter["generationNum"];
        outNeighborhoodSize = parameter["neighborhoodSize"];
        outDivisionsNumOfWeightVector = parameter["divisionsNumOfWeightVector"];
        outMigrationInterval = parameter["migrationInterval"];
        outCrossoverRate = parameter["crossoverRate"];
        outObjectivesNum = parameter["objectivesNum"];
        outIsAsync = parameter["isAsync"];
        outDecisionVariablesNums =
            parameter["decisionVariablesNums"].get<std::vector<int>>();
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

    std::vector<std::vector<int>> GatherDataTraffics(
        const std::vector<std::vector<int>>& dataTraffics) {
        std::vector<int> sendBuffer;
        int dataSize = dataTraffics[0].size();
        sendBuffer.reserve(dataSize * dataTraffics.size());
        for (const auto& dataTraffic : dataTraffics) {
            sendBuffer.insert(sendBuffer.end(), dataTraffic.begin(),
                              dataTraffic.end());
        }

        std::vector<int> receiveBuffer;
        std::vector<int> sizes;
        Gatherv(sendBuffer, rank, parallelSize, receiveBuffer, sizes);

        std::vector<std::vector<int>> allDataTraffics;
        allDataTraffics.reserve(receiveBuffer.size() / dataSize);
        if (rank == 0) {
            for (int rank = 0, count = 0; rank < parallelSize;
                 count += sizes[rank], rank++) {
                for (int j = count; j < count + sizes[rank]; j += dataSize) {
                    std::vector<int> traffic = {rank};
                    traffic.insert(traffic.end(), receiveBuffer.begin() + j,
                                   receiveBuffer.begin() + j + dataSize);
                    allDataTraffics.push_back(std::move(traffic));
                }
            }
        }

        return allDataTraffics;
    }

    std::vector<double> GatherExecutionTimes() {
        std::vector<double> globalExecutionTimes(executionTimes.size());
        for (std::size_t i = 0; i < executionTimes.size(); ++i) {
            MPI_Reduce(&executionTimes[i], &globalExecutionTimes[i], 1,
                       MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        }

        return globalExecutionTimes;
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
        int objectivesNum;
        bool isAsync;
        std::vector<int> decisionVariablesNums;
        std::vector<std::string> problemNames;
        std::string adjacencyListFileName;
        auto parameterFile = OpenInputFile(parameterFilePath);
        auto parameter = ReadParameters(
            parameterFile, trial, generationNum, neighborhoodSize,
            divisionsNumOfWeightVector, migrationInterval, crossoverRate,
            objectivesNum, isAsync, decisionVariablesNums, problemNames,
            adjacencyListFileName);
        parameterFile.close();

        if (decisionVariablesNums.size() != problemNames.size()) {
            throw std::invalid_argument(
                "The number of decision variables is not equal to the number "
                "of "
                "problems.");
        }

        // vectorの確保
        localObjectivesListHistory.reserve(trial);
        executionTimes.reserve(trial);

        // 出力ディレクトリの作成
        std::filesystem::path outputDirectoryPath;
        if (isAsync) {
            outputDirectoryPath =
                "out/data/" + moeadName + "Async" + "/" + GetTimestamp() + "/";
        } else {
            outputDirectoryPath =
                "out/data/" + moeadName + "Sync" + "/" + GetTimestamp() + "/";
        }
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

        for (std::size_t i = 0; i < problemNames.size(); i++) {
            const std::string& problemName = problemNames[i];
            int decisionVariablesNum = decisionVariablesNums[i];

            // 各種ディレクトリの作成
            const std::filesystem::path outputProblemDirectoryPath =
                outputDirectoryPath / problemName;
            const std::filesystem::path objectiveDirectoryPath =
                outputProblemDirectoryPath / "objective";
            const std::filesystem::path idealPointDirectoryPath =
                outputProblemDirectoryPath / "idealPoint";
            const std::filesystem::path igdDirectoryPath =
                outputProblemDirectoryPath / "igd";
            const std::filesystem::path dataTrafficsDirectoryPath =
                outputProblemDirectoryPath / "dataTraffics";
            RANK0(
                std::filesystem::create_directories(outputProblemDirectoryPath);
                std::filesystem::create_directories(objectiveDirectoryPath);
                std::filesystem::create_directories(idealPointDirectoryPath);
                std::filesystem::create_directories(igdDirectoryPath);
                std::filesystem::create_directories(dataTrafficsDirectoryPath);)

            // 実行時間ファイルの作成
            const std::filesystem::path executionTimesFilePath =
                outputProblemDirectoryPath / "executionTimes.csv";
            std::ofstream executionTimesFile;
            if (rank == 0) {
                executionTimesFile = OpenOutputFile(executionTimesFilePath);
                SetSignificantDigits(executionTimesFile, 9);
                WriteCsvLine(executionTimesFile, ElapsedTimeHeaders);
            }

            // moeadの構成クラスの作成
            std::shared_ptr<IProblem<double>> problem =
                CreateProblem(problemName, decisionVariablesNum, objectivesNum);
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
            std::vector<std::vector<double>> paretoFront;
            if (rank == 0) {
                auto paretoFrontFile = OpenInputFile(
                    "data/ground_truth/pareto_fronts/" + problemName + "-" +
                    std::to_string(problem->ObjectivesNum()) + ".csv");
                paretoFront = ReadCsv<double>(paretoFrontFile, false);
            }
            IGD indicator(paretoFront);

            RANK0(std::cout << "Problem: " << problemName << std::endl)

            for (int t = 0; t < trial; t++) {
                transitionOfIdealPoint.clear();
                localObjectivesListHistory.clear();
                executionTimes.clear();

                std::unique_ptr<IParallelMoead<double>> moead;
                if (moeadName == MoeadNames[0]) {
                    moead = std::make_unique<MpMoead<double>>(
                        generationNum, neighborhoodSize,
                        divisionsNumOfWeightVector, migrationInterval,
                        crossover, decomposition, mutation, problem, repair,
                        sampling, selection, isAsync);
                } else if (moeadName == MoeadNames[1]) {
                    moead = std::make_unique<MpMoeadIdealTopology<double>>(
                        generationNum, neighborhoodSize,
                        divisionsNumOfWeightVector, migrationInterval,
                        adjacencyListFileName, crossover, decomposition,
                        mutation, problem, repair, sampling, selection,
                        isAsync);
                } else if (moeadName == MoeadNames[2]) {
                    moead = std::make_unique<HalfMpMoead<double>>(
                        generationNum, neighborhoodSize,
                        divisionsNumOfWeightVector, migrationInterval,
                        adjacencyListFileName, crossover, decomposition,
                        mutation, problem, repair, sampling, selection,
                        isAsync);
                } else {
                    throw std::invalid_argument("Invalid moead name");
                }

                MPI_Barrier(MPI_COMM_WORLD);

                moead->Initialize();

                AddIdealPoint(moead->CurrentGeneration(),
                              decomposition->IdealPoint());
                localObjectivesListHistory.push_back(
                    moead->GetObjectivesList());
                executionTimes.push_back(moead->GetExecutionTime());

                MPI_Barrier(MPI_COMM_WORLD);

                while (!moead->IsEnd()) {
                    moead->Update();

                    AddIdealPoint(moead->CurrentGeneration(),
                                  decomposition->IdealPoint());
                    localObjectivesListHistory.push_back(
                        moead->GetObjectivesList());
                    executionTimes.push_back(moead->GetExecutionTime());
                }

                double elapsed = moead->GetExecutionTime();
                MPI_Barrier(MPI_COMM_WORLD);

                // 実行時間の出力
                double maxExecutionTime;
                MPI_Reduce(&elapsed, &maxExecutionTime, 1, MPI_DOUBLE, MPI_MAX,
                           0, MPI_COMM_WORLD);
                RANK0(std::cout << "Trial " << t + 1
                                << " Total execution time: " << maxExecutionTime
                                << " seconds" << std::endl;)
                RANK0(executionTimesFile << t + 1 << "," << maxExecutionTime
                                         << std::endl;)

                std::string fileName =
                    "trial_" + std::to_string(t + 1) + ".csv";

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
                auto globalExecutionTimes = GatherExecutionTimes();
                if (rank == 0) {
                    std::filesystem::path igdFilePath =
                        igdDirectoryPath / fileName;
                    std::ofstream igdFile = OpenOutputFile(igdFilePath);
                    SetSignificantDigits(igdFile);
                    std::vector<std::tuple<int, double, double>> igd;
                    for (int j = 0; j < objectivesListHistory.size(); j++) {
                        igd.push_back(std::make_tuple(
                            j, globalExecutionTimes[j],
                            indicator.Calculate(objectivesListHistory[j])));
                    }
                    // ヘッダーの書き込み
                    for (std::size_t j = 0; j < IgdHeader.size(); j++) {
                        igdFile << IgdHeader[j];
                        if (j != IgdHeader.size() - 1) {
                            igdFile << ",";
                        }
                    }
                    igdFile << std::endl;
                    // データの書き込み
                    for (const auto& [generation, time, igdValue] : igd) {
                        igdFile << generation << "," << time << "," << igdValue
                                << std::endl;
                    }
                }

                // データ量の出力
                auto sendDataTraffics = moead->GetDataTraffics();
                auto allDataTraffics = GatherDataTraffics(sendDataTraffics);
                if (rank == 0) {
                    std::filesystem::path dataTrafficsFilePath =
                        dataTrafficsDirectoryPath / fileName;
                    std::ofstream dataTrafficsFile =
                        OpenOutputFile(dataTrafficsFilePath);
                    WriteCsv(dataTrafficsFile, allDataTraffics,
                             DataTrafficsHeader);
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