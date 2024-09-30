#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include "Algorithms/Moead.h"
#include "Crossovers/BinomialCrossover.h"
#include "Decompositions/Tchebycheff.h"
#include "Mutations/PolynomialMutation.h"
#include "Problems/ZDT1.h"
#include "Repairs/SamplingRepair.h"
#include "Samplings/UniformRandomSampling.h"
#include "Selections/RandomSelection.h"

using namespace Eacpp;

int main(int argc, char* argv[]) {
    int generationNum = 500;
    int H = 299;
    int neighborNum = 21;

    if (argc == 2) {
        generationNum = std::stoi(argv[1]);
    } else if (argc == 3) {
        generationNum = std::stoi(argv[1]);
        H = std::stoi(argv[2]);
    } else if (argc == 4) {
        generationNum = std::stoi(argv[1]);
        H = std::stoi(argv[2]);
        neighborNum = std::stoi(argv[3]);
    } else if (argc >= 5) {
        std::cerr << "Usage: " << argv[0] << " [generationNum] [H] [neighborNum]" << std::endl;
        return 1;
    }

    auto problem = std::make_shared<ZDT1>();

    int decisionVariableNum = problem->decisionVariablesNum;
    int objectiveNum = problem->objectivesNum;
    std::vector<std::pair<double, double>> variableBound{problem->variableBound};

    auto crossover = std::make_shared<BinomialCrossover>(1.0, 0.5);
    auto decomposition = std::make_shared<Tchebycheff>(objectiveNum);
    auto mutation = std::make_shared<PolynomialMutation>(1.0 / decisionVariableNum, 20.0, variableBound);
    auto sampling = std::make_shared<UniformRandomSampling>(problem->variableBound.first, problem->variableBound.second);
    auto repair = std::make_shared<SamplingRepair<double>>(sampling);
    auto selection = std::make_shared<RandomSelection>();

    Moead<double> moead(generationNum, decisionVariableNum, objectiveNum, neighborNum, H, crossover, decomposition, mutation,
                        problem, repair, sampling, selection);

    auto start = std::chrono::system_clock::now();

    moead.Run();

    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Execution time: " << duration << "ms" << std::endl;

    std::filesystem::create_directories("out/data/");
    std::filesystem::create_directories("out/data/moead");
    std::ofstream ofs("out/data/moead/result.txt");
    for (const auto& objectives : moead.GetAllObjectives()) {
        for (const auto& value : objectives) {
            ofs << value << " ";
        }
        ofs << std::endl;
    }

    return 0;
}
