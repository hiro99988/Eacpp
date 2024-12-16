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
#include "Indicators.hpp"
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

/// @brief 更新されたときに理想点addをvecに追加する
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
    // パラメータ読み込み
    constexpr const char* ParameterFilePath = "data/inputs/benchmarks/parameter.json";
    auto parameterFile = OpenInputFile(ParameterFilePath);
    nlohmann::json parameter = nlohmann::json::parse(parameterFile);

    int trial = parameter["trial"];
    int generationNum = parameter["generationNum"];
    int neighborhoodSize = parameter["neighborhoodSize"];
    int divisionsNumOfWeightVector = parameter["divisionsNumOfWeightVector"];
    double crossoverRate = parameter["crossoverRate"];
    std::vector<std::string> problemNames = parameter["problems"];

    // 出力ディレクトリの作成
    const std::filesystem::path outputDirectoryPath = "out/data/Moead/" + GetTimestamp() + "/";
    std::filesystem::create_directories(outputDirectoryPath);

    // パラメータファイルのコピー
    std::filesystem::copy(ParameterFilePath, outputDirectoryPath / "parameter.json",
                          std::filesystem::copy_options::overwrite_existing);

    Stopwatch stopwatch;

    for (auto&& problemName : problemNames) {
        // 各種ディレクトリの作成
        const std::filesystem::path outputProblemDirectoryPath = outputDirectoryPath / problemName;
        const std::filesystem::path objectiveDirectoryPath = outputProblemDirectoryPath / "objective";
        const std::filesystem::path idealPointDirectoryPath = outputProblemDirectoryPath / "idealPoint";
        const std::filesystem::path igdDirectoryPath = outputProblemDirectoryPath / "igd";
        std::filesystem::create_directories(outputProblemDirectoryPath);
        std::filesystem::create_directories(objectiveDirectoryPath);
        std::filesystem::create_directories(idealPointDirectoryPath);
        std::filesystem::create_directories(igdDirectoryPath);

        // 実行時間ファイルの作成
        constexpr std::array<const char*, 2> ExecutionTimesHeader = {"trial", "time(s)"};
        const std::filesystem::path executionTimesFilePath = outputProblemDirectoryPath / "executionTimes.csv";
        std::ofstream executionTimesFile = OpenOutputFile(executionTimesFilePath);
        SetSignificantDigits(executionTimesFile, 9);
        WriteCsvLine(executionTimesFile, ExecutionTimesHeader);

        // moeadの構成クラスの作成
        std::shared_ptr<IProblem<double>> problem = std::move(Reflection<IProblem<double>>::Create(problemName));
        auto crossover = std::make_shared<SimulatedBinaryCrossover>(crossoverRate, problem->VariableBounds());
        auto decomposition = std::make_shared<Tchebycheff>();
        auto mutation = std::make_shared<PolynomialMutation>(1.0 / problem->DecisionVariablesNum(), problem->VariableBounds());
        auto sampling = std::make_shared<RealRandomSampling>(problem->VariableBounds());
        auto repair = std::make_shared<RealRandomRepair>(problem);
        auto selection = std::make_shared<RandomSelection>();

        // ヘッダの作成
        std::vector<std::string> objectiveHeader;
        for (int i = 0; i < problem->ObjectivesNum(); i++) {
            objectiveHeader.push_back("objective" + std::to_string(i + 1));
        }
        std::vector<std::string> idealPointHeader = {"generation"};
        for (int i = 0; i < problem->ObjectivesNum(); i++) {
            idealPointHeader.push_back("objective" + std::to_string(i + 1));
        }
        constexpr std::array<const char*, 2> IgdHeader = {"generation", "igd"};

        // インディケータの作成
        std::ifstream paretoFrontFile("data/ground_truth/pareto_fronts/" + problemName + ".csv");
        auto paretoFront = ReadCsv<double>(paretoFrontFile, false);
        IGD indicator(paretoFront);

        std::cout << "Problem: " << problemName << std::endl;

        for (int i = 0; i < trial; i++) {
            Moead<double> moead(generationNum, neighborhoodSize, divisionsNumOfWeightVector, crossover, decomposition, mutation,
                                problem, repair, sampling, selection);

            std::vector<std::pair<int, Eigen::ArrayXd>> transitionOfIdealPoint;
            std::vector<std::vector<Eigen::ArrayXd>> populations;
            populations.reserve(generationNum + 1);

            stopwatch.Restart();

            moead.Initialize();

            stopwatch.Stop();

            AddIdealPoint(moead.CurrentGeneration(), decomposition->IdealPoint(), transitionOfIdealPoint);
            populations.push_back(moead.GetObjectivesList());

            while (!moead.IsEnd()) {
                stopwatch.Start();

                moead.Update();

                stopwatch.Stop();

                AddIdealPoint(moead.CurrentGeneration(), decomposition->IdealPoint(), transitionOfIdealPoint);
                populations.push_back(moead.GetObjectivesList());
            }

            std::cout << "Trial " << i + 1 << " Total execution time: " << stopwatch.Elapsed() << " seconds" << std::endl;

            // 目的関数値の出力
            std::filesystem::path objectiveFilePath = objectiveDirectoryPath / ("trial_" + std::to_string(i + 1) + ".csv");
            std::ofstream objectiveFile = OpenOutputFile(objectiveFilePath);
            SetSignificantDigits(objectiveFile);
            WriteCsv(objectiveFile, moead.GetObjectivesList(), objectiveHeader);

            // 理想点の出力
            std::filesystem::path idealPointFilePath = idealPointDirectoryPath / ("trial_" + std::to_string(i + 1) + ".csv");
            std::ofstream idealPointFile = OpenOutputFile(idealPointFilePath);
            SetSignificantDigits(idealPointFile);
            WriteCsv(idealPointFile, transitionOfIdealPoint, idealPointHeader);

            // IGDの出力
            const std::filesystem::path igdFilePath = igdDirectoryPath / ("trial_" + std::to_string(i + 1) + ".csv");
            std::ofstream igdFile = OpenOutputFile(igdFilePath);
            SetSignificantDigits(igdFile);
            std::vector<std::pair<int, double>> igd;
            for (int j = 0; j < populations.size(); j++) {
                igd.push_back(std::make_pair(j, indicator.Calculate(populations[j])));
            }
            WriteCsv(igdFile, igd, IgdHeader);

            // 実行時間の出力
            executionTimesFile << i + 1 << "," << stopwatch.Elapsed() << std::endl;
        }
    }

    return 0;
}