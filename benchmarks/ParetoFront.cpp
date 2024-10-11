#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "Problems/IBenchmark.h"
#include "Problems/Problems.h"
#include "Reflections/Reflection.h"

void PrintUsage(const char* programPath) {
    std::cerr << "Usage: " << programPath << " <pointsNum: int>" << std::endl;
    std::cerr << "pointsNum - number of points to generate for each problem." << std::endl;
}

int ConvertToInt(const char* str) {
    try {
        return std::stoi(str);
    } catch (const std::invalid_argument& e) {
        std::cerr << "Invalid argument: " << str << " is not an integer." << std::endl;
        std::exit(1);
    } catch (const std::out_of_range& e) {
        std::cerr << "Out of range: " << str << " is too large." << std::endl;
        std::exit(1);
    }
}

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

std::unique_ptr<Eacpp::IBenchmark> CreateProblem(const std::string& className) {
    std::unique_ptr<Eacpp::IBenchmark> problem = Eacpp::Reflection<Eacpp::IBenchmark>::Create(className);
    if (!problem) {
        std::cerr << "Error: Could not create problem " << className << std::endl;
        std::exit(1);
    }
    return problem;
}

std::string CreateOutFileName(const std::string& className, int pointsNum) {
    return className + "_" + std::to_string(pointsNum) + ".csv";
}

void WriteParetoFront(std::ofstream& outfile, const std::vector<Eigen::ArrayXd>& paretoFront) {
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

int main(int argc, char** argv) {
    constexpr const char* outDirPath = "data/ground_truth/pareto_fronts/";
    constexpr const char* targetProblemsFilePath = "data/inputs/benchmarks/TargetProblems.txt";

    if (argc != 2) {
        PrintUsage(argv[0]);
        return 1;
    }

    int pointsNum = ConvertToInt(argv[1]);

    std::ifstream infile = OpenInputFile(targetProblemsFilePath);

    const std::filesystem::path outDir(outDirPath);
    CreateDirectories(outDir);

    std::string className;
    while (infile >> className) {
        std::unique_ptr<Eacpp::IBenchmark> problem = CreateProblem(className);

        std::vector<Eigen::ArrayXd> paretoFront = problem->GenerateParetoFront(pointsNum);

        const std::filesystem::path outFilePath = outDir / CreateOutFileName(className, pointsNum);
        std::ofstream outfile = OpenOutputFile(outFilePath);

        WriteParetoFront(outfile, paretoFront);
    }

    return 0;
}