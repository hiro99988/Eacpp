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
#include "Rng/Rng.h"
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
    int decisionVariablesNum;
    int objectivesNum;
};

void from_json(const nlohmann::json& j, Problem& prob) {
    j.at("name").get_to(prob.name);
    if (j.contains("level")) j.at("level").get_to(prob.level);
    if (j.contains("decisionVariablesNum"))
        j.at("decisionVariablesNum").get_to(prob.decisionVariablesNum);
    if (j.contains("objectivesNum"))
        j.at("objectivesNum").get_to(prob.objectivesNum);
}

class ParallelMoeadBenchmark {
   public:
    constexpr static const char* DefaultParameterFilePath =
        "data/inputs/benchmarks/hpaParameter.json";
    constexpr static const char* UtopiaNadirFilePath =
        "extern/hpa/utopia_and_nadir_points/n=4/"
        "utopia_nadir.json";
    constexpr static std::array<const char*, 3> MoeadNames = {"MP-MOEAD",
                                                              "MP-MOEAD-NO"};
    constexpr static std::array<const char*, 4> ElapsedTimeHeaders = {
        "trial", "initialization_time_s", "execution_time_s",
        "communication_time_s"};
    constexpr static std::array<const char*, 6> DataTrafficsHeader = {
        "rank",          "generation",
        "send_times",    "total_send_sata_traffic",
        "receive_times", "total_receive_data_traffic"};

    // key: 目的数, H
    const std::unordered_map<int, int> divisionsNumOfWeightVectors = {
        {2, 79}, {3, 14}, {4, 8}, {5, 6}, {6, 5}, {9, 4}};
    // key: level, 評価回数
    const std::unordered_map<int, int> evaluationsNums = {
        {0, 72000}, {1, 72000}, {2, 216000}};

    int rank;
    int parallelSize;
    std::vector<Eigen::ArrayXd> transitionOfIdealPoint;
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

            // TODO: MPI_Allgatherv
            // を使ってこの段階で全てのランクに全ての目的関数値を分散させる
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
                        outObjectivesListHistory[j].reserve(
                            objectivesListHistory[j].size());
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
        // aがbを支配するかどうかを判定するラムダ関数
        auto dominates = [](const std::vector<double>& a,
                            const std::vector<double>& b) -> bool {
            bool strictlyBetter = false;
            for (std::size_t i = 0; i < a.size(); ++i) {
                if (a[i] > b[i]) {
                    return false;
                } else if (a[i] < b[i]) {
                    strictlyBetter = true;
                }
            }
            return strictlyBetter;
        };

        // 全ての解を結合
        std::vector<std::pair<int, std::vector<double>>> merged = newObjectives;
        merged.insert(merged.end(), existingNonDominated.begin(),
                      existingNonDominated.end());
        int mergedSize = static_cast<int>(merged.size());

        // 全プロセスでサイズを共有
        MPI_Bcast(&mergedSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // 全プロセスで解の次元数を共有
        int dimension = 0;
        if (rank == 0 && mergedSize > 0) {
            dimension = static_cast<int>(merged[0].second.size());
        }
        MPI_Bcast(&dimension, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // 解のみを buffer に集約
        std::vector<double> buffer;
        if (rank == 0) {
            buffer.reserve(mergedSize * dimension);
            for (auto& m : merged) {
                buffer.insert(buffer.end(), m.second.begin(), m.second.end());
            }
        }
        buffer.resize(mergedSize * dimension);
        // 全プロセスにブロードキャスト
        MPI_Bcast(buffer.data(), static_cast<int>(buffer.size()), MPI_DOUBLE, 0,
                  MPI_COMM_WORLD);

        // 受信後，各プロセスは buffer を解のベクトルに変換
        std::vector<std::vector<double>> mergedSolutions;
        if (rank != 0) {
            mergedSolutions.reserve(mergedSize);
            for (int i = 0; i < mergedSize; ++i) {
                mergedSolutions.emplace_back(
                    buffer.begin() + i * dimension,
                    buffer.begin() + (i + 1) * dimension);
            }
        }

        // 各プロセスに部分範囲を割り当て
        // rankごとに [start, end) のインデックスを担当
        int workload = CalculateNodeWorkload(mergedSize, rank, parallelSize);
        int start = CalculateNodeStartIndex(mergedSize, rank, parallelSize);
        int end = start + workload;

        // 割り当てられた範囲において自分が支配されている（=1）か判定
        std::vector<int> isDominatedLocal(mergedSize, 0);
        if (rank == 0) {
            for (int i = start; i < end; ++i) {
                for (int j = 0; j < mergedSize; ++j) {
                    if (i == j) continue;
                    if (dominates(merged[j].second, merged[i].second)) {
                        isDominatedLocal[i] = 1;
                        break;
                    }
                }
            }
        } else {
            for (int i = start; i < end; ++i) {
                for (int j = 0; j < mergedSize; ++j) {
                    if (i == j) continue;
                    if (dominates(mergedSolutions[j], mergedSolutions[i])) {
                        isDominatedLocal[i] = 1;
                        break;
                    }
                }
            }
        }

        // Rank 0に集約
        std::vector<int> isDominatedGlobal;
        if (rank == 0) {
            isDominatedGlobal.resize(mergedSize, 0);
        }
        MPI_Reduce(isDominatedLocal.data(), isDominatedGlobal.data(),
                   mergedSize, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

        // Rank 0が非支配解を構築して返す
        if (rank == 0) {
            std::vector<std::pair<int, std::vector<double>>> nonDominated;
            nonDominated.reserve(mergedSize);
            for (int i = 0; i < mergedSize; ++i) {
                if (isDominatedGlobal[i] == 0) {
                    nonDominated.push_back(std::move(merged[i]));
                }
            }
            nonDominated.shrink_to_fit();
            return nonDominated;
        } else {
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

    std::vector<std::tuple<int, int, std::vector<double>>>
    GatherTransitionOfIdealPoint() {
        std::vector<double> sendBuffer;
        int dataSize = transitionOfIdealPoint[0].size();
        sendBuffer.reserve(dataSize * transitionOfIdealPoint.size());
        for (const auto& idealPoint : transitionOfIdealPoint) {
            sendBuffer.insert(sendBuffer.end(), idealPoint.begin(),
                              idealPoint.end());
        }

        std::vector<double> receiveBuffer;
        std::vector<int> sizes;
        Gatherv(sendBuffer, rank, parallelSize, receiveBuffer, sizes);

        // [[rank, gen, idealPoint], ...]
        std::vector<std::tuple<int, int, std::vector<double>>>
            transitionOfIdealPointList;
        if (rank == 0) {
            transitionOfIdealPointList.reserve(receiveBuffer.size() / dataSize);
            for (int rank = 0, count = 0; rank < parallelSize;
                 count += sizes[rank], rank++) {
                int gen = 0;
                for (int j = count; j < count + sizes[rank]; j += dataSize) {
                    std::vector<double> idealPoint(
                        receiveBuffer.begin() + j,
                        receiveBuffer.begin() + j + dataSize);
                    transitionOfIdealPointList.emplace_back(
                        rank, gen, std::move(idealPoint));
                    gen++;
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

    std::vector<double> GatherMaxTimes(const std::vector<double>& times) {
        std::vector<double> globalTimes(times.size());
        for (std::size_t i = 0; i < times.size(); ++i) {
            MPI_Reduce(&times[i], &globalTimes[i], 1, MPI_DOUBLE, MPI_MAX, 0,
                       MPI_COMM_WORLD);
        }

        return globalTimes;
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

        for (auto&& problem : problems) {
            RANK0(std::cout << "Problem: " << problem.name << std::endl)

            // 問題ディレクトリの作成
            std::filesystem::path problemDirectoryPath =
                outputDirectoryPath / problem.name;
            RANK0(std::filesystem::create_directories(problemDirectoryPath);)

            // 問題名の修正．"-"以降を削除．問題作成時の名前用
            // 例: "HPA201-0" -> "HPA201"
            std::string officialName = problem.name;
            std::size_t pos = officialName.find('-');
            if (pos != std::string::npos) {
                officialName = officialName.substr(0, pos);
            }
            // 問題名の二つ目の"-"以降を削除．パレートフロントファイル名用
            // 例: "DTLZ1-3-10" -> "DTLZ1-3"
            std::string paretoFrontFileName = problem.name;
            if (paretoFrontFileName.compare(0, 3, "ZDT") == 0) {
                paretoFrontFileName = officialName;
            } else {
                std::size_t firstDashPos = paretoFrontFileName.find('-');
                if (firstDashPos != std::string::npos) {
                    std::size_t secondDashPos =
                        paretoFrontFileName.find('-', firstDashPos + 1);
                    if (secondDashPos != std::string::npos) {
                        paretoFrontFileName =
                            paretoFrontFileName.substr(0, secondDashPos);
                    }
                }
            }

            // HPA問題かどうか確認
            bool isHpaProblem = false;
            if (officialName.compare(0, 3, "HPA") == 0) {
                isHpaProblem = true;
            }

            // 問題クラスの作成
            std::shared_ptr<IProblem<double>> problem_ptr;
            if (isHpaProblem) {
                problem_ptr = std::make_unique<Hpa>(
                    hpaModule, officialName.c_str(), 4, problem.level);
            } else {
                problem_ptr =
                    CreateProblem(officialName, problem.decisionVariablesNum,
                                  problem.objectivesNum);
            }

            int divisionsNumOfWeightVector =
                divisionsNumOfWeightVectors.at(problem_ptr->ObjectivesNum());
            int evaluationsNum = evaluationsNums.at(problem.level);
            int generationsNum =
                evaluationsNum /
                CalculatePopulationSize(divisionsNumOfWeightVector,
                                        problem_ptr->ObjectivesNum());
            RANK0(std::cout << "generationsNum: " << generationsNum
                            << std::endl;)

            // HPA問題の場合，正規化のための utopia, nadir 点を取得
            std::vector<double> utopia;
            std::vector<double> nadir;
            if (isHpaProblem) {
                auto utopiaNadirJson = OpenInputFile(UtopiaNadirFilePath);
                nlohmann::json utopiaNadir =
                    nlohmann::json::parse(utopiaNadirJson);
                utopia = utopiaNadir[problem.name]["utopia"]
                             .get<std::vector<double>>();
                nadir = utopiaNadir[problem.name]["nadir"]
                            .get<std::vector<double>>();
                utopiaNadirJson.close();
            }

            // インディケータの作成
            std::vector<std::vector<double>> paretoFront;
            if (rank == 0) {
                std::filesystem::path paretoFrontFilePath =
                    isHpaProblem
                        ? std::filesystem::path("extern/hpa/pareto_fronts/n=4/")
                              .append(paretoFrontFileName + ".csv")
                        : std::filesystem::path(
                              "data/ground_truth/pareto_fronts/")
                              .append(paretoFrontFileName + ".csv");
                auto paretoFrontFile = OpenInputFile(paretoFrontFilePath);
                paretoFront = ReadCsv<double>(paretoFrontFile, true, true);
            }
            // パレートフロントも正規化
            if (isHpaProblem) {
                for (auto&& objectives : paretoFront) {
                    for (std::size_t i = 0; i < objectives.size(); i++) {
                        objectives[i] = (objectives[i] - utopia[i]) /
                                        (nadir[i] - utopia[i]);
                    }
                }
            }
            IGDPlus indicator(paretoFront);
            std::vector<std::string> igdHeader = {"generation",
                                                  "execution_time_s"};
            {
                // 指標の名前を小文字に変換して追加
                std::string indicatorName = indicator.Name();
                std::transform(indicatorName.begin(), indicatorName.end(),
                               indicatorName.begin(), ::tolower);
                igdHeader.push_back(indicatorName);
            }

            for (auto&& algorithm : algorithms) {
                // MP-MOEA/D-NOのobjと問題の目的数が一致するか確認
                if (algorithm.name.compare(0, std::string(MoeadNames[1]).size(),
                                           MoeadNames[1]) == 0) {
                    if (std::find(algorithm.obj.begin(), algorithm.obj.end(),
                                  problem_ptr->ObjectivesNum()) ==
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
                const std::filesystem::path elapsedTimesFilePath =
                    outputAlgorithmDirectoryPath / "elapsedTimes.csv";
                std::ofstream elapsedTimesFile;
                if (rank == 0) {
                    elapsedTimesFile = OpenOutputFile(elapsedTimesFilePath);
                    SetSignificantDigits(elapsedTimesFile, 9);
                    WriteCsvLine(elapsedTimesFile, ElapsedTimeHeaders);
                }

                // moeadの構成クラスの作成
                auto crossover = std::make_shared<SimulatedBinaryCrossover>(
                    crossoverRate, problem_ptr->VariableBounds());
                auto decomposition = std::make_shared<Tchebycheff>();
                auto mutation = std::make_shared<PolynomialMutation>(
                    1.0 / problem_ptr->DecisionVariablesNum(),
                    problem_ptr->VariableBounds());
                auto repair = std::make_shared<RealRandomRepair>(problem_ptr);
                auto selection = std::make_shared<RandomSelection>();

                // 目的関数ファイルのヘッダの作成
                std::vector<std::string> objectiveHeader = {"rank"};
                for (int i = 0; i < problem_ptr->ObjectivesNum(); i++) {
                    objectiveHeader.push_back("objective" +
                                              std::to_string(i + 1));
                }
                // 理想点ファイルのヘッダの作成
                std::vector<std::string> idealPointHeader = {"rank",
                                                             "generation"};
                for (int i = 0; i < problem_ptr->ObjectivesNum(); i++) {
                    idealPointHeader.push_back("objective" +
                                               std::to_string(i + 1));
                }

                RANK0(std::cout << "Algorithm: " << algorithm.name << std::endl)

                for (int t = 0; t < trial; t++) {
                    transitionOfIdealPoint.clear();
                    localObjectivesListHistory.clear();
                    executionTimes.clear();

                    auto sampling = rank == 0
                                        ? std::make_shared<RealRandomSampling>(
                                              problem_ptr->VariableBounds(),
                                              std::make_shared<Rng>(t))
                                        : std::make_shared<RealRandomSampling>(
                                              problem_ptr->VariableBounds());

                    std::unique_ptr<IParallelMoead<double>> moead;
                    if (algorithm.name == MoeadNames[0]) {
                        moead = std::make_unique<MpMoead<double>>(
                            generationsNum, neighborhoodSize,
                            divisionsNumOfWeightVector, migrationInterval,
                            crossover, decomposition, mutation, problem_ptr,
                            repair, sampling, selection, algorithm.isAsync);
                    } else if (algorithm.name.compare(
                                   0, std::string(MoeadNames[1]).size(),
                                   MoeadNames[1]) == 0) {
                        moead = std::make_unique<MpMoeadIdealTopology<double>>(
                            generationsNum, neighborhoodSize,
                            divisionsNumOfWeightVector, migrationInterval,
                            algorithm.adjacencyListFileName, crossover,
                            decomposition, mutation, problem_ptr, repair,
                            sampling, selection, algorithm.isAsync);
                    } else {
                        throw std::invalid_argument("Invalid moead name");
                    }

                    MPI_Barrier(MPI_COMM_WORLD);

                    moead->Initialize();

                    transitionOfIdealPoint.push_back(
                        decomposition->IdealPoint());
                    localObjectivesListHistory.push_back(
                        moead->GetObjectivesList());
                    executionTimes.push_back(0.);

                    MPI_Barrier(MPI_COMM_WORLD);

                    while (!moead->IsEnd()) {
                        moead->Update();

                        transitionOfIdealPoint.push_back(
                            decomposition->IdealPoint());
                        localObjectivesListHistory.push_back(
                            moead->GetObjectivesList());
                        executionTimes.push_back(moead->GetExecutionTime());
                    }

                    double initializationTime = moead->GetInitializationTime();
                    double executionTime = moead->GetExecutionTime();
                    double communicationTime = moead->GetCommunicationTime();
                    MPI_Barrier(MPI_COMM_WORLD);

                    // 実行時間の出力
                    double maxInitializationTime;
                    double maxExecutionTime;
                    double maxCommunicationTime;
                    MPI_Reduce(&initializationTime, &maxInitializationTime, 1,
                               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                    MPI_Reduce(&executionTime, &maxExecutionTime, 1, MPI_DOUBLE,
                               MPI_MAX, 0, MPI_COMM_WORLD);
                    MPI_Reduce(&communicationTime, &maxCommunicationTime, 1,
                               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                    RANK0(std::cout
                              << "Trial " << t + 1 << " Initialization time: "
                              << maxInitializationTime
                              << " Total execution time: " << maxExecutionTime
                              << " Communication time: " << maxCommunicationTime
                              << std::endl;)
                    RANK0(elapsedTimesFile
                              << t + 1 << "," << maxInitializationTime << ","
                              << maxExecutionTime << "," << maxCommunicationTime
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
                        // ヘッダーの書き込み
                        WriteCsvLine(idealPointFile, idealPointHeader);
                        // データの書き込み
                        for (const auto& [rank, gen, idealPoint] :
                             transitionOfIdealPointList) {
                            idealPointFile << rank << "," << gen << ",";
                            for (std::size_t i = 0; i < idealPoint.size();
                                 ++i) {
                                // 正規化
                                auto value = isHpaProblem
                                                 ? (idealPoint[i] - utopia[i]) /
                                                       (nadir[i] - utopia[i])
                                                 : idealPoint[i];
                                idealPointFile << value;
                                if (i != idealPoint.size() - 1) {
                                    idealPointFile << ",";
                                }
                            }
                            idealPointFile << std::endl;
                        }
                    }

                    // 目的関数値の正規化
                    if (isHpaProblem) {
                        for (auto& objectivesList :
                             localObjectivesListHistory) {
                            for (auto& objectives : objectivesList) {
                                for (std::size_t i = 0; i < objectives.size();
                                     i++) {
                                    objectives[i] =
                                        (objectives[i] - utopia[i]) /
                                        (nadir[i] - utopia[i]);
                                }
                            }
                        }
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
                    auto globalExecutionTimes = GatherMaxTimes(executionTimes);
                    if (rank == 0) {
                        // IGD出力ファイルの作成
                        std::filesystem::path igdFilePath =
                            igdDirectoryPath / fileName;
                        std::ofstream igdFile = OpenOutputFile(igdFilePath);
                        SetSignificantDigits(igdFile);

                        // IGDの計算
                        std::vector<std::tuple<int, double, double>> igd;
                        for (int j = 0; j < objectivesListHistory.size(); j++) {
                            igd.push_back(std::make_tuple(
                                j, globalExecutionTimes[j],
                                indicator.Calculate(objectivesListHistory[j])));
                        }
                        // ヘッダーの書き込み
                        for (std::size_t j = 0; j < igdHeader.size(); j++) {
                            igdFile << igdHeader[j];
                            if (j != igdHeader.size() - 1) {
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