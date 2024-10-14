#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include "Algorithms/Moead.h"
#include "Crossovers/SimulatedBinaryCrossover.h"
#include "Decompositions/Tchebycheff.h"
#include "Mutations/PolynomialMutation.h"
#include "Problems/ZDT6.h"
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

    auto problem = std::make_shared<ZDT6>();

    std::pair<double, double> variableBound{problem->VariableBound()};
    std::vector<std::pair<double, double>> variableBounds{variableBound};

    auto crossover = std::make_shared<SimulatedBinaryCrossover>(0.9);
    auto decomposition = std::make_shared<Tchebycheff>();
    auto mutation = std::make_shared<PolynomialMutation>(1.0 / problem->DecisionVariablesNum(), 20.0, variableBounds);
    auto sampling = std::make_shared<UniformRandomSampling>(variableBound.first, variableBound.second);
    auto repair = std::make_shared<SamplingRepair<double>>(sampling);
    auto selection = std::make_shared<RandomSelection>();

    Moead<double> moead(generationNum, neighborNum, H, crossover, decomposition, mutation, problem, repair, sampling,
                        selection);

    auto start = std::chrono::system_clock::now();

    moead.Run();

    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Execution time: " << duration << "ms" << std::endl;

    return 0;
}
