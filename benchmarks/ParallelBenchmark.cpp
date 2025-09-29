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

#include "Algorithms/MpMoead.h"
#include "Crossovers/SimulatedBinaryCrossover.h"
#include "Decompositions/Tchebycheff.h"
#include "Indicators.hpp"
#include "Mutations/PolynomialMutation.h"
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

namespace eacpp {

struct Algorithm {
    std::string name;
    bool isAsync;
    int migrationInterval;
};

void from_json(const nlohmann::json& j, Algorithm& alg) {
    j.at("name").get_to(alg.name);
    j.at("isAsync").get_to(alg.isAsync);
    j.at("migrationInterval").get_to(alg.migrationInterval);
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

struct Division {
    int obj;
    int value;
};

void from_json(const nlohmann::json& j, Division& div) {
    j.at("obj").get_to(div.obj);
    j.at("value").get_to(div.value);
}

struct Generation {
    int level;
    int value;
};

void from_json(const nlohmann::json& j, Generation& gen) {
    j.at("level").get_to(gen.level);
    j.at("value").get_to(gen.value);
}

class ParallelMoeadBenchmark {
   public:
    constexpr static const char* DefaultParameterFilePath =
        "data/inputs/ParallelParameter.json";
    constexpr static std::array<const char*, 3> MoeadNames = {"MP-MOEAD"};
    constexpr static std::array<const char*, 3> ElapsedTimeHeaders = {
        "trial", "initialization_time_s", "execution_time_s"};

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
        for (int i = 0; i < 3; ++i) {
            auto problem = std::make_shared<ZDT1>();
            auto crossover = std::make_shared<SimulatedBinaryCrossover>(
                0.9, problem->VariableBounds());
            auto decomposition = std::make_shared<Tchebycheff>();
            auto mutation = std::make_shared<PolynomialMutation>(
                1.0 / problem->DecisionVariablesNum(),
                problem->VariableBounds());
            auto sampling =
                std::make_shared<RealRandomSampling>(problem->VariableBounds());
            auto repair = std::make_shared<RealRandomRepair>(problem);
            auto selection = std::make_shared<RandomSelection>();

            auto moead = MpMoead<double>(10000, 21, 299, 1, crossover,
                                         decomposition, mutation, problem,
                                         repair, sampling, selection, true);

            moead.Run();

            MPI_Barrier(MPI_COMM_WORLD);
            ReleaseIsend(parallelSize, MPI_DOUBLE);
        }
    }

    void GatherObjectivesListHistory(
        std::vector<std::vector<std::vector<double>>>& outObjectivesListHistory,
        std::vector<std::pair<int, std::vector<double>>>& finalObjectivesList) {
        if (rank == 0) {
            outObjectivesListHistory.reserve(localObjectivesListHistory.size());
        }

        for (std::size_t i = 0; i < localObjectivesListHistory.size(); i++) {
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

    void GatherNonDominatedSolutionsList(
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

            std::sort(nonDominated.begin(), nonDominated.end(),
                      [](const auto& lhs, const auto& rhs) {
                          return lhs.second < rhs.second;
                      });
            nonDominated.erase(
                std::unique(nonDominated.begin(), nonDominated.end(),
                            [](const auto& lhs, const auto& rhs) {
                                return lhs.second == rhs.second;
                            }),
                nonDominated.end());

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

        bool isIndicatorCalculatedUsingNds =
            parameter["isIndicatorCalculatedUsingNds"];
        int trial = parameter["trial"];
        int neighborhoodSize = parameter["neighborhoodSize"];
        double crossoverRate = parameter["crossoverRate"];
        std::vector<Division> divisionsNumOfWeightVectors =
            parameter.at("divisionsNumOfWeightVectors")
                .get<std::vector<Division>>();
        std::vector<Generation> generationsNums =
            parameter.at("generationsNums").get<std::vector<Generation>>();
        std::vector<Algorithm> algorithms =
            parameter.at("algorithms").get<std::vector<Algorithm>>();
        std::vector<Problem> problems =
            parameter.at("problems").get<std::vector<Problem>>();

        parameterFile.close();

        // vectorの確保
        localObjectivesListHistory.reserve(trial);
        executionTimes.reserve(trial);

        RANK0(std::cout << "Start warmup" << std::endl)
        Warmup();
        RANK0(std::cout << "End warmup" << std::endl)

        // アルゴリズム名の先頭がMoeadNamesに含まれているか確認
        for (auto&& algorithm : algorithms) {
            bool found = false;
            for (auto&& name : MoeadNames) {
                if (algorithm.name.rfind(name, 0) == 0) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                throw std::invalid_argument(
                    "Algorithm name must start with MOEAD: " + algorithm.name);
            }
        }
        // アルゴリズム名に重複がないか確認
        for (std::size_t i = 0; i < algorithms.size(); i++) {
            for (std::size_t j = i + 1; j < algorithms.size(); j++) {
                if (algorithms[i].name == algorithms[j].name) {
                    throw std::invalid_argument("Duplicate algorithm name: " +
                                                algorithms[i].name);
                }
            }
        }

// 出力ディレクトリの作成
#ifdef BENCHMARK_OUTPUT_DIR
        std::filesystem::path outputDirectoryPath =
            BENCHMARK_OUTPUT_DIR + GetTimestamp();
#else
        std::filesystem::path outputDirectoryPath =
            "out/data/" + GetTimestamp();
#endif
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
            // 例: "ZDT1-30" -> "ZDT1"
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

            // 問題クラスの作成
            std::shared_ptr<IProblem<double>> problem_ptr =
                CreateProblem(officialName, problem.decisionVariablesNum,
                              problem.objectivesNum);

            int divisionsNumOfWeightVector;
            int generationsNum;
            for (auto&& div : divisionsNumOfWeightVectors) {
                if (div.obj == problem_ptr->ObjectivesNum()) {
                    divisionsNumOfWeightVector = div.value;
                    break;
                }
            }
            for (auto&& gen : generationsNums) {
                if (gen.level == problem.level) {
                    generationsNum = gen.value;
                    break;
                }
            }

            // インディケータの作成
            std::vector<std::vector<double>> paretoFront;
            if (rank == 0) {
                std::filesystem::path paretoFrontFilePath =
                    std::filesystem::path("data/ground_truth/pareto_fronts/")
                        .append(paretoFrontFileName + ".csv");
                auto paretoFrontFile = OpenInputFile(paretoFrontFilePath);
                paretoFront = ReadCsv<double>(paretoFrontFile, true, true);
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
                // 各種ディレクトリの作成
                const std::filesystem::path outputAlgorithmDirectoryPath =
                    problemDirectoryPath / algorithm.name;
                const std::filesystem::path objectiveDirectoryPath =
                    outputAlgorithmDirectoryPath / "objective";
                const std::filesystem::path idealPointDirectoryPath =
                    outputAlgorithmDirectoryPath / "idealPoint";
                const std::filesystem::path igdDirectoryPath =
                    outputAlgorithmDirectoryPath / "igdPlus";
                if (rank == 0) {
                    std::filesystem::create_directories(
                        outputAlgorithmDirectoryPath);
                    std::filesystem::create_directories(objectiveDirectoryPath);
                    std::filesystem::create_directories(
                        idealPointDirectoryPath);
                    std::filesystem::create_directories(igdDirectoryPath);
                }

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
                    if (algorithm.name.rfind(MoeadNames[0], 0) == 0) {
                        moead = std::make_unique<MpMoead<double>>(
                            generationsNum, neighborhoodSize,
                            divisionsNumOfWeightVector,
                            algorithm.migrationInterval, crossover,
                            decomposition, mutation, problem_ptr, repair,
                            sampling, selection, algorithm.isAsync);
                    } else {
                        throw std::invalid_argument("Invalid moead name: " +
                                                    algorithm.name);
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
                    MPI_Barrier(MPI_COMM_WORLD);

                    // 実行時間の出力
                    double maxInitializationTime;
                    double maxExecutionTime;
                    MPI_Reduce(&initializationTime, &maxInitializationTime, 1,
                               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                    MPI_Reduce(&executionTime, &maxExecutionTime, 1, MPI_DOUBLE,
                               MPI_MAX, 0, MPI_COMM_WORLD);
                    RANK0(std::cout
                              << "Trial " << t + 1 << " Initialization time: "
                              << maxInitializationTime
                              << " Total execution time: " << maxExecutionTime
                              << std::endl;)
                    RANK0(elapsedTimesFile << t + 1 << ","
                                           << maxInitializationTime << ","
                                           << maxExecutionTime << std::endl;)

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
                                idealPointFile << idealPoint[i];
                                if (i != idealPoint.size() - 1) {
                                    idealPointFile << ",";
                                }
                            }
                            idealPointFile << std::endl;
                        }
                    }

                    // 目的関数値の出力
                    std::vector<std::vector<std::vector<double>>>
                        objectivesListHistory;
                    std::vector<std::pair<int, std::vector<double>>>
                        finalObjectivesList;
                    if (isIndicatorCalculatedUsingNds) {
                        GatherNonDominatedSolutionsList(objectivesListHistory,
                                                        finalObjectivesList);
                    } else {
                        GatherObjectivesListHistory(objectivesListHistory,
                                                    finalObjectivesList);
                    }
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

                        // isIndicatorCalculatedUsingNds == false
                        // の時IGDが最小の世代の目的関数値を記録する
                        double minIgd = std::numeric_limits<double>::max();
                        std::size_t minIgdIndex = 0;

                        // IGDの計算
                        std::vector<std::tuple<int, double, double>> igd;
                        for (int j = 0; j < objectivesListHistory.size(); j++) {
                            double igdValue =
                                indicator.Calculate(objectivesListHistory[j]);
                            igd.push_back(std::make_tuple(
                                j, globalExecutionTimes[j], igdValue));

                            if (!isIndicatorCalculatedUsingNds) {
                                if (minIgd > igdValue) {
                                    minIgd = igdValue;
                                    minIgdIndex = j;
                                }
                            }
                        }
                        // ヘッダーの書き込み
                        WriteCsvLine(igdFile, igdHeader);
                        // データの書き込み
                        for (const auto& [generation, time, igdValue] : igd) {
                            igdFile << generation << "," << time << ","
                                    << igdValue << std::endl;
                        }

                        if (!isIndicatorCalculatedUsingNds) {
                            // 最小IGDの世代の目的関数値を記録する
                            const std::filesystem::path
                                minObjectiveDirectoryPath =
                                    outputAlgorithmDirectoryPath /
                                    "minObjective";
                            std::filesystem::create_directories(
                                minObjectiveDirectoryPath);
                            std::filesystem::path minObjectiveFilePath =
                                minObjectiveDirectoryPath / fileName;
                            std::ofstream minObjectiveFile =
                                OpenOutputFile(minObjectiveFilePath);
                            SetSignificantDigits(minObjectiveFile);
                            std::vector<std::string> minObjectiveHeader(
                                objectiveHeader.begin() + 1,
                                objectiveHeader.end());
                            WriteCsv(minObjectiveFile,
                                     objectivesListHistory[minIgdIndex],
                                     minObjectiveHeader);
                        }
                    }

                    MPI_Barrier(MPI_COMM_WORLD);
                    ReleaseIsend(parallelSize, MPI_DOUBLE);
                }
            }
        }
    }
};

}  // namespace eacpp

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto benchmark = eacpp::ParallelMoeadBenchmark();
    benchmark.Run();

    MPI_Finalize();
    return 0;
}