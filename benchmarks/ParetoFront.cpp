#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "Problems/BenchmarkReflection.h"
#include "Problems/IBenchmark.h"
#include "Problems/Problems.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <integer>" << std::endl;
        return 1;
    }

    int pointsNum;
    try {
        pointsNum = std::stoi(argv[1]);
    } catch (const std::invalid_argument& e) {
        std::cerr << "Invalid argument: " << argv[1] << " is not an integer." << std::endl;
        return 1;
    } catch (const std::out_of_range& e) {
        std::cerr << "Out of range: " << argv[1] << " is too large." << std::endl;
        return 1;
    }

    const std::filesystem::path outDir = "data/ground_truth/pareto_fronts/";
    const std::filesystem::path targetFile = "data/inputs/benchmarks/TargetProblems.txt";

    std::ifstream infile(targetFile);
    if (!infile) {
        std::cerr << "Error: Could not open " << targetFile << std::endl;
        return 1;
    }

    std::filesystem::create_directories(outDir);

    std::string className;
    while (infile >> className) {
        std::unique_ptr<Eacpp::IBenchmark> problem = Eacpp::BenchmarkReflection::Create(className);
        if (!problem) {
            std::cerr << "Error: Could not create problem " << className << std::endl;
            return 1;
        }

        std::vector<Eigen::ArrayXd> paretoFront = problem->GenerateParetoFront(pointsNum);

        const std::filesystem::path outFile = outDir / (className + "_" + std::to_string(pointsNum) + ".csv");
        std::ofstream outfile(outFile);
        if (!outfile) {
            std::cerr << "Error: Could not create or open " << outFile << std::endl;
            return 1;
        }

        for (const auto& point : paretoFront) {
            for (int i = 0; i < point.size(); ++i) {
                outfile << point(i);
                if (i < point.size() - 1) {
                    outfile << ",";
                }
            }
            outfile << std::endl;
        }
    }

    return 0;
}