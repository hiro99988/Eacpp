#include <mpi.h>
#include <pybind11/embed.h>

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

#include "Algorithms/HalfMpMoead.hpp"
#include "Algorithms/MpMoead.h"
#include "Algorithms/MpMoeadIdealTopology.h"
#include "Algorithms/NtMoead.h"
#include "Algorithms/OneNtMoead.hpp"
#include "Crossovers/SimulatedBinaryCrossover.h"
#include "Decompositions/Tchebycheff.h"
#include "Indicators.hpp"
#include "Mutations/PolynomialMutation.h"
#include "Problems/Hpa.hpp"
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

struct Algorithm {
    std::string name;
    bool isAsync;
    std::string adjacencyListFileName;  // オプション項目
    std::vector<int> obj;               // オプション項目
};

void from_json(const nlohmann::json& j, Algorithm& alg) {
    j.at("name").get_to(alg.name);
    j.at("isAsync").get_to(alg.isAsync);
    // オプション項目は存在する場合のみ取得
    if (j.contains("adjacencyListFileName"))
        j.at("adjacencyListFileName").get_to(alg.adjacencyListFileName);
    if (j.contains("obj")) j.at("obj").get_to(alg.obj);
}

struct Problem {
    std::string name;
    int level;
};

void from_json(const nlohmann::json& j, Problem& prob) {
    j.at("name").get_to(prob.name);
    j.at("level").get_to(prob.level);
}

class ParallelMoeadBenchmark {
   public:
    constexpr static const char* DefaultParameterFilePath =
        "data/inputs/benchmarks/hpaParameter.json";
    constexpr static std::array<const char*, 3> MoeadNames = {"MP-MOEAD",
                                                              "MP-MOEAD-NO"};
    constexpr static std::array<const char*, 2> ExecutionTimesHeader = {
        "trial", "time(s)"};
    constexpr static std::array<const char*, 3> IgdHeader = {
        "generation", "executionTime(s)", "igd"};
    constexpr static std::array<const char*, 6> DataTrafficsHeader = {
        "rank",         "generation",
        "sendTimes",    "totalSendDataTraffic",
        "receiveTimes", "totalReceiveDataTraffic"};

    // key: 目的数, H
    const std::unordered_map<int, int> divisionsNumOfWeightVectors = {
        {2, 79}, {3, 14}, {4, 8}, {5, 6}, {6, 5}, {9, 4}};
    // key: level, 評価回数
    const std::unordered_map<int, int> evaluationsNums = {
        {0, 72000}, {1, 72000}, {2, 216000}};

    int rank;
    int parallelSize;
    std::vector<std::pair<int, Eigen::ArrayXd>> transitionOfIdealPoint;
    std::vector<std::vector<Eigen::ArrayXd>> localObjectivesListHistory;
    std::vector<double> executionTimes;

    ParallelMoeadBenchmark() {}

    void InitializeMpi() {
        int initialized;
        MPI_Initialized(&initialized);
        if (!initialized) {
            MPI_Init(nullptr, nullptr);
        }
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &parallelSize);
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
        std::vector<std::vector<std::pair<int, std::vector<double>>>>
            objectivesListHistory;
        if (rank == 0) {
            objectivesListHistory.reserve(localObjectivesListHistory.size());
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

            std::vector<std::pair<int, std::vector<double>>> objectivesList;
            if (rank == 0) {
                // 世代 i の目的関数値
                for (int j = 0, count = 0; j < parallelSize;
                     count += sizes[j], j++) {
                    for (int k = 0; k < sizes[j]; k += dataSize) {
                        std::vector<double> objectives(
                            receiveBuffer.begin() + count + k,
                            receiveBuffer.begin() + count + k + dataSize);

                        objectivesList.push_back(
                            std::make_pair(j, std::move(objectives)));
                    }
                }
            }

            // 目的関数値の非支配解を計算
            auto nonDominated = ComputeNonDominatedSolutions(
                objectivesList,
                i == 0 || rank != 0
                    ? std::vector<std::pair<int, std::vector<double>>>{}
                    : objectivesListHistory.back());

            if (rank == 0) {
                if (i == localObjectivesListHistory.size() - 1) {
                    // 最終世代の非支配解を保存
                    finalObjectivesList = nonDominated;
                    objectivesListHistory.push_back(std::move(nonDominated));
                    // outObjectivesListHistoryにobjectiveListHistoryのstd::vectorだけ保存
                    outObjectivesListHistory.resize(
                        objectivesListHistory.size());
                    for (std::size_t j = 0; j < objectivesListHistory.size();
                         ++j) {
                        for (auto&& objectivesList : objectivesListHistory[j]) {
                            outObjectivesListHistory[j].push_back(
                                std::move(objectivesList.second));
                        }
                    }
                } else {
                    objectivesListHistory.push_back(std::move(nonDominated));
                }
            }
        }
    }

    std::vector<std::pair<int, std::vector<double>>>
    ComputeNonDominatedSolutions(
        const std::vector<std::pair<int, std::vector<double>>>& newObjectives,
        const std::vector<std::pair<int, std::vector<double>>>&
            existingNonDominated) {
        // Lambda to check if solution a dominates solution b
        auto dominates = [](const std::vector<double>& a,
                            const std::vector<double>& b) -> bool {
            bool strictlyBetter = false;
            for (std::size_t i = 0; i < a.size(); ++i) {
                if (a[i] > b[i]) {  // if any objective is worse, 'a' does not
                                    // dominate 'b'
                    return false;
                } else if (a[i] < b[i]) {
                    strictlyBetter = true;
                }
            }
            return strictlyBetter;
        };

        // Merge solutions
        std::vector<std::pair<int, std::vector<double>>> merged = newObjectives;
        merged.insert(merged.end(), existingNonDominated.begin(),
                      existingNonDominated.end());
        int N = static_cast<int>(merged.size());

        // ------------------------
        // 1) 全プロセスでサイズNを共有
        // ------------------------
        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // 解をシリアライズしてブロードキャスト (rank 0 以外は受信領域を確保)
        // 各解は int + vector<double>
        // のため、以下の例では単純に連続double配列化します （元々rank
        // 0が全解を持っている想定）
        int dimension = 0;
        if (rank == 0 && N > 0) {
            dimension = static_cast<int>(merged[0].second.size());
        }
        MPI_Bcast(&dimension, 1, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<double> buffer;
        buffer.reserve(N * (dimension + 1));  // +1はpair<int, ...>のintの格納用
        if (rank == 0) {
            for (auto& m : merged) {
                // 先にpair<int, ...>のint
                buffer.push_back(static_cast<double>(m.first));
                // 次に解ベクトル
                buffer.insert(buffer.end(), m.second.begin(), m.second.end());
            }
        }
        buffer.resize(N * (dimension + 1));
        // 全プロセスにブロードキャスト
        MPI_Bcast(buffer.data(), static_cast<int>(buffer.size()), MPI_DOUBLE, 0,
                  MPI_COMM_WORLD);

        // 受信後、各プロセスはmergedを再構築
        if (rank != 0) {
            merged.resize(N);
            for (int i = 0; i < N; ++i) {
                int offset = i * (dimension + 1);
                int info = static_cast<int>(buffer[offset]);
                std::vector<double> sol(dimension);
                for (int d = 0; d < dimension; ++d) {
                    sol[d] = buffer[offset + 1 + d];
                }
                merged[i] = std::make_pair(info, std::move(sol));
            }
        }

        // ------------------------
        // 2) 各プロセスに部分範囲を割り当て
        // ------------------------
        // rankごとに [start, end) のインデックスを担当
        int chunkSize = (N + parallelSize - 1) / parallelSize;
        int start = rank * chunkSize;
        int end = std::min(start + chunkSize, N);

        // この範囲について「自分が支配されているか」を判定 (isDominatedLocal[i]
        // = 1 or 0)
        std::vector<int> isDominatedLocal(N, 0);

        for (int i = start; i < end; ++i) {
            for (int j = 0; j < N; ++j) {
                if (i == j) continue;
                if (dominates(merged[j].second, merged[i].second)) {
                    isDominatedLocal[i] = 1;
                    break;
                }
            }
        }

        // ------------------------
        // 3) Rank 0に集約
        // ------------------------
        std::vector<int> isDominatedGlobal(N, 0);
        MPI_Reduce(isDominatedLocal.data(), isDominatedGlobal.data(), N,
                   MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

        // ------------------------
        // 4) Rank 0が非支配解を構築して返す
        // ------------------------
        if (rank == 0) {
            std::vector<std::pair<int, std::vector<double>>> nonDominated;
            nonDominated.reserve(N);
            for (int i = 0; i < N; ++i) {
                if (isDominatedGlobal[i] == 0) {
                    nonDominated.push_back(merged[i]);
                }
            }
            return nonDominated;
        } else {
            // Rank 0以外は空（または必要があれば同期して同一データを返す）
            return {};
        }
    }

    // 逐次的に非支配解を計算する関数
    // std::vector<std::pair<int, std::vector<double>>>
    // ComputeNonDominatedSolutions(
    //     const std::vector<std::pair<int, std::vector<double>>>&
    //     newObjectives, const std::vector<std::pair<int,
    //     std::vector<double>>>&
    //         existingNonDominated) {
    //     // 全ての解を結合
    //     std::vector<std::pair<int, std::vector<double>>> allSolutions;
    //     allSolutions.reserve(newObjectives.size() +
    //                          existingNonDominated.size());
    //     allSolutions.insert(allSolutions.end(), newObjectives.begin(),
    //                         newObjectives.end());
    //     allSolutions.insert(allSolutions.end(), existingNonDominated.begin(),
    //                         existingNonDominated.end());

    //     // 辞書順に昇順ソート
    //     std::sort(allSolutions.begin(), allSolutions.end(),
    //               [](const std::pair<int, std::vector<double>>& lhs,
    //                  const std::pair<int, std::vector<double>>& rhs) {
    //                   return std::lexicographical_compare(
    //                       lhs.second.begin(), lhs.second.end(),
    //                       rhs.second.begin(), rhs.second.end());
    //               });

    //     // 暫定非支配集合
    //     std::vector<std::pair<int, std::vector<double>>> nonDominated;
    //     nonDominated.reserve(allSolutions.size());

    //     // dominates関数
    //     auto dominates = [](const std::vector<double>& a,
    //                         const std::vector<double>& b) {
    //         bool strictlyBetter = false;
    //         for (std::size_t i = 0; i < a.size(); ++i) {
    //             if (a[i] > b[i]) return false;
    //             if (a[i] < b[i]) strictlyBetter = true;
    //         }
    //         return strictlyBetter;
    //     };

    //     // 暫定非支配集合に対し、新たな点が支配されていないかを判定
    //     for (auto&& current : allSolutions) {
    //         bool dominatedByExisting = false;
    //         // 既存非支配集合をチェック
    //         for (auto& nd : nonDominated) {
    //             if (dominates(nd.second, current.second)) {
    //                 dominatedByExisting = true;
    //                 break;
    //             }
    //         }
    //         // 非支配ならリストに追加し、既存リスト側が支配されていれば除去
    //         if (!dominatedByExisting) {
    //             // 逆にcurrentがndを支配しているなら、そのndを削除
    //             nonDominated.erase(
    //                 std::remove_if(nonDominated.begin(), nonDominated.end(),
    //                                [&](auto& nd) {
    //                                    return dominates(current.second,
    //                                                     nd.second);
    //                                }),
    //                 nonDominated.end());
    //             nonDominated.push_back(std::move(current));
    //         }
    //     }

    //     return nonDominated;
    // }

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

    int CalculatePopulationSize(int divisionsNumOfWeightVector,
                                int objectivesNum) const {
        int n = divisionsNumOfWeightVector + objectivesNum - 1;
        int r = objectivesNum - 1;
        return Combination(n, r);
    }

    void Run() {
        InitializeMpi();

        // パラメータ読み込み
        auto parameterFile = OpenInputFile(DefaultParameterFilePath);
        nlohmann::json parameter = nlohmann::json::parse(parameterFile);

        int trial = parameter["trial"];
        int neighborhoodSize = parameter["neighborhoodSize"];
        int migrationInterval = parameter["migrationInterval"];
        double crossoverRate = parameter["crossoverRate"];
        bool idealPointMigration = parameter["idealPointMigration"];
        std::vector<Algorithm> algorithms =
            parameter.at("algorithms").get<std::vector<Algorithm>>();
        std::vector<Problem> problems =
            parameter.at("problems").get<std::vector<Problem>>();

        parameterFile.close();

        // vectorの確保
        localObjectivesListHistory.reserve(trial);
        executionTimes.reserve(trial);

        // pythonインタープリターの初期化
        pybind11::scoped_interpreter guard{};
        pybind11::module hpaModule =
            pybind11::module::import("extern.hpa.hpa.problem");

        RANK0(std::cout << "Start warmup" << std::endl)
        Warmup();
        RANK0(std::cout << "End warmup" << std::endl)

        // アルゴリズム名が正しいか確認
        for (auto&& algorithm : algorithms) {
            bool valid = false;
            for (const auto& prefix : MoeadNames) {
                if (algorithm.name.compare(0, std::string(prefix).size(),
                                           prefix) == 0) {
                    valid = true;
                    break;
                }
            }
            if (!valid) {
                throw std::invalid_argument("Invalid algorithm name: " +
                                            algorithm.name);
            }
        }

        // 出力ディレクトリの作成
        std::filesystem::path outputDirectoryPath =
            "out/data/" + GetTimestamp();
        RANK0(std::filesystem::create_directories(outputDirectoryPath);)
        // パラメータファイルのコピー
        if (rank == 0) {
            parameter["parallelSize"] = parallelSize;
            std::string parameterString = parameter.dump(4);
            auto parameterOutputFile =
                OpenOutputFile(outputDirectoryPath / "parameter.json");
            parameterOutputFile << parameterString;
            parameterOutputFile.close();
        }

        for (auto&& hpaProblem : problems) {
            RANK0(std::cout << "Problem: " << hpaProblem.name << std::endl)

            // 問題ディレクトリの作成
            std::filesystem::path problemDirectoryPath =
                outputDirectoryPath / hpaProblem.name;
            RANK0(std::filesystem::create_directories(problemDirectoryPath);)

            // 問題名の修正．"-"以降を削除
            // 例: "HPA201-0" -> "HPA201"
            std::string hpaName = hpaProblem.name;
            std::size_t pos = hpaName.find('-');
            if (pos != std::string::npos) {
                hpaName = hpaName.substr(0, pos);
            }

            // HPA問題の作成
            std::shared_ptr<IProblem<double>> problem = std::make_unique<Hpa>(
                hpaModule, hpaName.c_str(), 4, hpaProblem.level);
            int divisionsNumOfWeightVector =
                divisionsNumOfWeightVectors.at(problem->ObjectivesNum());
            int evaluationsNum = evaluationsNums.at(hpaProblem.level);
            int generationsNum =
                evaluationsNum /
                CalculatePopulationSize(divisionsNumOfWeightVector,
                                        problem->ObjectivesNum());
            RANK0(std::cout << "generationsNum: " << generationsNum
                            << std::endl;)

            for (auto&& algorithm : algorithms) {
                // MP-MOEA/D-NOのobjとhpaの目的数が一致するか確認
                if (algorithm.name.compare(0, std::string(MoeadNames[1]).size(),
                                           MoeadNames[1]) == 0) {
                    if (std::find(algorithm.obj.begin(), algorithm.obj.end(),
                                  problem->ObjectivesNum()) ==
                        algorithm.obj.end()) {
                        continue;
                    }
                }

                // 各種ディレクトリの作成
                const std::filesystem::path outputAlgorithmDirectoryPath =
                    problemDirectoryPath / algorithm.name;
                const std::filesystem::path objectiveDirectoryPath =
                    outputAlgorithmDirectoryPath / "objective";
                const std::filesystem::path idealPointDirectoryPath =
                    outputAlgorithmDirectoryPath / "idealPoint";
                const std::filesystem::path igdDirectoryPath =
                    outputAlgorithmDirectoryPath / "igd";
                const std::filesystem::path dataTrafficsDirectoryPath =
                    outputAlgorithmDirectoryPath / "dataTraffics";
                RANK0(
                    std::filesystem::create_directories(
                        outputAlgorithmDirectoryPath);
                    std::filesystem::create_directories(objectiveDirectoryPath);
                    std::filesystem::create_directories(
                        idealPointDirectoryPath);
                    std::filesystem::create_directories(igdDirectoryPath);
                    std::filesystem::create_directories(
                        dataTrafficsDirectoryPath);)

                // 実行時間ファイルの作成
                const std::filesystem::path executionTimesFilePath =
                    outputAlgorithmDirectoryPath / "executionTimes.csv";
                std::ofstream executionTimesFile;
                if (rank == 0) {
                    executionTimesFile = OpenOutputFile(executionTimesFilePath);
                    SetSignificantDigits(executionTimesFile, 9);
                    WriteCsvLine(executionTimesFile, ExecutionTimesHeader);
                }

                // moeadの構成クラスの作成
                auto crossover = std::make_shared<SimulatedBinaryCrossover>(
                    crossoverRate, problem->VariableBounds());
                auto decomposition = std::make_shared<Tchebycheff>();
                auto mutation = std::make_shared<PolynomialMutation>(
                    1.0 / problem->DecisionVariablesNum(),
                    problem->VariableBounds());
                auto sampling = std::make_shared<RealRandomSampling>(
                    problem->VariableBounds());
                auto repair = std::make_shared<RealRandomRepair>(problem);
                auto selection = std::make_shared<RandomSelection>();

                // ヘッダの作成
                std::vector<std::string> objectiveHeader = {"rank"};
                for (int i = 0; i < problem->ObjectivesNum(); i++) {
                    objectiveHeader.push_back("objective" +
                                              std::to_string(i + 1));
                }
                std::vector<std::string> idealPointHeader = {"rank",
                                                             "generation"};
                for (int i = 0; i < problem->ObjectivesNum(); i++) {
                    idealPointHeader.push_back("objective" +
                                               std::to_string(i + 1));
                }

                // インディケータの作成
                std::vector<std::vector<double>> paretoFront;
                if (rank == 0) {
                    auto paretoFrontFile =
                        OpenInputFile("extern/hpa/igd_reference_points/n=4/" +
                                      hpaProblem.name + ".csv");
                    paretoFront = ReadCsv<double>(paretoFrontFile, true);
                }
                IGD indicator(paretoFront);

                RANK0(std::cout << "Algorithm: " << algorithm.name << std::endl)

                for (int t = 0; t < trial; t++) {
                    transitionOfIdealPoint.clear();
                    localObjectivesListHistory.clear();
                    executionTimes.clear();

                    std::unique_ptr<IParallelMoead<double>> moead;
                    if (algorithm.name == MoeadNames[0]) {
                        moead = std::make_unique<MpMoead<double>>(
                            generationsNum, neighborhoodSize,
                            divisionsNumOfWeightVector, migrationInterval,
                            crossover, decomposition, mutation, problem, repair,
                            sampling, selection, idealPointMigration,
                            algorithm.isAsync);
                    } else if (algorithm.name.compare(
                                   0, std::string(MoeadNames[1]).size(),
                                   MoeadNames[1]) == 0) {
                        moead = std::make_unique<MpMoeadIdealTopology<double>>(
                            generationsNum, neighborhoodSize,
                            divisionsNumOfWeightVector, migrationInterval,
                            algorithm.adjacencyListFileName, crossover,
                            decomposition, mutation, problem, repair, sampling,
                            selection, algorithm.isAsync);
                    } else {
                        throw std::invalid_argument("Invalid moead name");
                    }

                    MPI_Barrier(MPI_COMM_WORLD);

                    moead->Initialize();

                    AddIdealPoint(moead->CurrentGeneration(),
                                  decomposition->IdealPoint());
                    localObjectivesListHistory.push_back(
                        moead->GetObjectivesList());
                    executionTimes.push_back(moead->GetElapsedTime());

                    MPI_Barrier(MPI_COMM_WORLD);

                    while (!moead->IsEnd()) {
                        moead->Update();

                        AddIdealPoint(moead->CurrentGeneration(),
                                      decomposition->IdealPoint());
                        localObjectivesListHistory.push_back(
                            moead->GetObjectivesList());
                        executionTimes.push_back(moead->GetElapsedTime());
                    }

                    double elapsed = moead->GetElapsedTime();
                    MPI_Barrier(MPI_COMM_WORLD);

                    // 実行時間の出力
                    double maxExecutionTime;
                    MPI_Reduce(&elapsed, &maxExecutionTime, 1, MPI_DOUBLE,
                               MPI_MAX, 0, MPI_COMM_WORLD);
                    RANK0(std::cout
                              << "Trial " << t + 1 << " Total execution time: "
                              << maxExecutionTime << " seconds" << std::endl;)
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
                            igdFile << generation << "," << time << ","
                                    << igdValue << std::endl;
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
    }
};

}  // namespace Eacpp

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto benchmark = Eacpp::ParallelMoeadBenchmark();
    benchmark.Run();

    MPI_Finalize();
    return 0;
}