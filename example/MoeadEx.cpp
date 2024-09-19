#include <array>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include "Algorithms/Moead.h"
#include "Crossovers/BinomialCrossover.h"
#include "Decompositions/Tchebycheff.h"
#include "Mutations/PolynomialMutation.h"
#include "Problems/ZDT1.h"
#include "Samplings/UniformRandomSampling.h"
#include "Selections/RandomSelection.h"

using namespace Eacpp;

int main(int argc, char* argv[]) {
    int generationNum = 500;
    int H = 299;
    int neighborNum = 20;

    if (argc == 2) {
        generationNum = std::stoi(argv[1]);
    } else if (argc == 3) {
        generationNum = std::stoi(argv[1]);
        H = std::stoi(argv[2]);
    } else if (argc == 4) {
        generationNum = std::stoi(argv[1]);
        H = std::stoi(argv[2]);
        neighborNum = std::stoi(argv[4]);
    } else if (argc >= 5) {
        std::cerr << "Usage: " << argv[0] << " [generationNum] [H] [neighborNum]" << std::endl;
        return 1;
    }

    auto problem = std::make_shared<ZDT1>();

    int decisionVariableNum = problem->decisionNum;
    int objectiveNum = problem->objectiveNum;
    std::vector<std::array<double, 2>> variableBounds = {problem->variableBound};

    auto crossover = std::make_shared<BinomialCrossover>(1.0, 0.5);
    auto decomposition = std::make_shared<Tchebycheff>();
    auto mutation = std::make_shared<PolynomialMutation>(1.0 / decisionVariableNum, 20.0, variableBounds);
    auto sampling = std::make_shared<UniformRandomSampling>(problem->variableBound[0], problem->variableBound[1]);
    auto selection = std::make_shared<RandomSelection>();

    Moead<double> moead(generationNum, decisionVariableNum, objectiveNum, neighborNum, H, crossover, decomposition, mutation,
                        problem, sampling, selection);
    moead.Run();

    std::ofstream ofs("out/data/result.txt");
    for (const auto& set : moead.objectiveSets) {
        for (const auto& value : set) {
            ofs << value << " ";
        }
        ofs << std::endl;
    }

    return 0;
}
