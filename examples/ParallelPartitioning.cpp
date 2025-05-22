#include <mpi.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <toml.hpp>
#include <vector>

#include "Algorithms/MoeadInitializer.h"
#include "Algorithms/SimulatedAnnealing.hpp"
#include "Graph/SimpleGraph.hpp"
#include "Problems/WeightVectorPartitioning.hpp"
#include "Utils/Utils.h"

using namespace Eacpp;

struct Parameter {
    int divisionsNumOfWeightVector;
    int objectivesNum;
    int neighborhoodSize;
    int parallelSize;
    double initialTemperature;
    double coolingRate;
    double minTemperature;
    int maxIterationsPerTemp;
    long long maxTotalIterations;
    int maxStagnantIterations;
};

std::tuple<double, typename SA<SolutionType>::Result, std::vector<std::size_t>,
           std::size_t, std::size_t>
RunSimulatedAnnealing(const Parameter& param) {
    bool verbose = false;

    // 初期解を生成
    MoeadInitializer moeadInitializer;
    auto vectorDivisions = moeadInitializer.GenerateWeightVectorDivisions(
        param.divisionsNumOfWeightVector, param.objectivesNum);
    auto initialSolution = moeadInitializer.LinearPartitioning(
        param.parallelSize, param.divisionsNumOfWeightVector, vectorDivisions);

    // SA構成クラスの生成
    auto problem = std::make_unique<PartitioningProblem>(
        param.divisionsNumOfWeightVector, param.objectivesNum,
        param.neighborhoodSize, param.parallelSize);
    auto neighborGen = std::make_unique<NeighborGen>();

    // 初期解の評価
    double initialObjective = problem->ComputeObjective(initialSolution);
    // Simulated Annealingのインスタンスを生成
    SA<SolutionType> sa(initialSolution, std::move(problem),
                        std::move(neighborGen), param.initialTemperature,
                        param.coolingRate, param.minTemperature,
                        param.maxIterationsPerTemp, param.maxTotalIterations,
                        param.maxStagnantIterations, verbose);
    // Simulated Annealingの実行
    auto result = sa.Run();

    // 最良解のグラフを生成
    auto problemForGraph = std::make_unique<PartitioningProblem>(
        param.divisionsNumOfWeightVector, param.objectivesNum,
        param.neighborhoodSize, param.parallelSize);
    auto graph = problemForGraph->GenerateGraph(result.best);

    // グラフの次数
    auto degrees = graph.Degrees();
    auto maxDegree = graph.MaxDegree();
    auto minDegree = graph.MinDegree();

    return {initialObjective, result, degrees, maxDegree, minDegree};
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<Parameter> allParams;

    try {
        const auto data = toml::parse("data/inputs/partitioningParameter.toml");
        const auto& params_from_toml =
            toml::find<toml::array>(data, "parameters");

        for (const auto& p_toml : params_from_toml) {
            Parameter p_struct;
            p_struct.divisionsNumOfWeightVector =
                toml::find<int>(p_toml, "divisionsNumOfWeightVector");
            p_struct.objectivesNum = toml::find<int>(p_toml, "objectivesNum");
            p_struct.neighborhoodSize =
                toml::find<int>(p_toml, "neighborhoodSize");
            p_struct.parallelSize = toml::find<int>(p_toml, "parallelSize");
            p_struct.initialTemperature =
                toml::find<double>(p_toml, "initialTemperature");
            p_struct.coolingRate = toml::find<double>(p_toml, "coolingRate");
            p_struct.minTemperature =
                toml::find<double>(p_toml, "minTemperature");
            p_struct.maxIterationsPerTemp =
                toml::find<int>(p_toml, "maxIterationsPerTemp");
            p_struct.maxTotalIterations =
                toml::find<long long>(p_toml, "maxTotalIterations");
            p_struct.maxStagnantIterations =
                toml::find<int>(p_toml, "maxStagnantIterations");
            allParams.push_back(p_struct);
        }
    } catch (const toml::exception& e) {
        std::cerr << "TOML parsing error (rank 0): " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    } catch (const std::out_of_range& e) {
        std::cerr << "TOML key not found (rank 0): " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::filesystem::path baseOutputDir =
        "out/data";  // Base directory for all results

    for (int i = rank; i < allParams.size(); i += size) {
        const Parameter& currentParam = allParams[i];

        std::cout << "Rank " << rank << " processing parameter set " << i
                  << std::endl;

        auto [initialObjective, saResult, degrees, maxDegree, minDegree] =
            RunSimulatedAnnealing(currentParam);

        // Create output directory
        std::string timestamp = GetTimestamp();
        std::stringstream dirNameStream;
        dirNameStream << currentParam.divisionsNumOfWeightVector << "-"
                      << currentParam.objectivesNum << "-"
                      << currentParam.neighborhoodSize << "-"
                      << currentParam.parallelSize << "-" << timestamp;
        std::filesystem::path currentOutputDirectory =
            baseOutputDir / dirNameStream.str();

        try {
            std::filesystem::create_directories(currentOutputDirectory);
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Rank " << rank << ": Error creating directory "
                      << currentOutputDirectory << ": " << e.what()
                      << std::endl;
            continue;  // Skip to next parameter set or handle error
        }

        // Write results to TOML file
        toml::table resultToml;
        resultToml["initialObjective"] = initialObjective;
        resultToml["finalObjective"] = saResult.objective;
        resultToml["iterationAtBest"] = saResult.iterationAtBest;
        resultToml["totalIterations"] = saResult.totalIterations;
        resultToml["temperatureAtBest"] = saResult.temperatureAtBest;
        resultToml["finalTemperature"] = saResult.finalTemperature;
        resultToml["totalExecutionTimeSeconds"] =
            saResult.totalExecutionTimeSeconds;
        resultToml["executionTimeAtBestSeconds"] =
            saResult.executionTimeAtBestSeconds;
        resultToml["maxDegree"] = maxDegree;
        resultToml["minDegree"] = minDegree;

        toml::array degrees_toml_array;
        for (const auto& deg : degrees) {
            degrees_toml_array.push_back(static_cast<long long>(
                deg));  // TOML likes long long for integers
        }
        resultToml["degrees"] = degrees_toml_array;

        toml::table inputParamsToml;
        inputParamsToml["divisionsNumOfWeightVector"] =
            currentParam.divisionsNumOfWeightVector;
        inputParamsToml["objectivesNum"] = currentParam.objectivesNum;
        inputParamsToml["neighborhoodSize"] = currentParam.neighborhoodSize;
        inputParamsToml["parallelSize"] = currentParam.parallelSize;
        inputParamsToml["initialTemperature"] = currentParam.initialTemperature;
        inputParamsToml["coolingRate"] = currentParam.coolingRate;
        inputParamsToml["minTemperature"] = currentParam.minTemperature;
        inputParamsToml["maxIterationsPerTemp"] =
            currentParam.maxIterationsPerTemp;
        inputParamsToml["maxTotalIterations"] = currentParam.maxTotalIterations;
        inputParamsToml["maxStagnantIterations"] =
            currentParam.maxStagnantIterations;
        resultToml["inputParameters"] = inputParamsToml;

        std::ofstream resultsFile(currentOutputDirectory / "results.toml");
        if (resultsFile.is_open()) {
            toml::value data_to_format(
                resultToml);  // Create a toml::value from the table
            resultsFile << toml::format(
                data_to_format);  // Format the toml::value
            resultsFile.close();
        } else {
            std::cerr << "Rank " << rank
                      << ": Failed to open results.toml for writing in "
                      << currentOutputDirectory << std::endl;
        }

        // Write saResult.best to CSV file
        std::ofstream solutionFile(currentOutputDirectory / "solution.csv");
        if (solutionFile.is_open()) {
            for (size_t p_idx = 0; p_idx < saResult.best.size(); ++p_idx) {
                const auto& matrix_partition =
                    saResult
                        .best[p_idx];  // This is std::vector<std::vector<int>>
                for (const auto& row :
                     matrix_partition)  // This is std::vector<int>
                {
                    for (size_t j = 0; j < row.size(); ++j) {
                        solutionFile << row[j]
                                     << (j == row.size() - 1 ? "" : ",");
                    }
                    solutionFile << "\n";
                }
                if (p_idx < saResult.best.size() - 1) {
                    solutionFile << "\n";  // Add a blank line to separate
                                           // matrix partitions
                }
            }
            solutionFile.close();
        } else {
            std::cerr << "Rank " << rank
                      << ": Failed to open solution.csv for writing in "
                      << currentOutputDirectory << std::endl;
        }
        if (rank == 0) {
            std::cout << "Rank " << rank
                      << " finished processing parameter set " << i
                      << ". Results in " << currentOutputDirectory << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}