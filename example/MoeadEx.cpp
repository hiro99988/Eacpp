#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>

#include "Algorithms/Moead.h"
#include "Crossovers/BinomialCrossover.h"
#include "Decompositions/Tchebycheff.h"
#include "Mutations/PolynomialMutation.h"
#include "Problems/LZ1.h"
#include "Samplings/UniformRandomSampling.h"
#include "Selections/RandomSelection.h"

using namespace Eacpp;

int main(int argc, char* argv[]) {
    int generationNum = 500;
    int H = 299;
    int decisionVariableNum = 30;
    int neighborNum = 20;
    if (argc == 2) {
        generationNum = std::stoi(argv[1]);
    } else if (argc == 3) {
        generationNum = std::stoi(argv[1]);
        H = std::stoi(argv[2]);
    } else if (argc == 4) {
        generationNum = std::stoi(argv[1]);
        H = std::stoi(argv[2]);
        decisionVariableNum = std::stoi(argv[3]);
    } else if (argc == 5) {
        generationNum = std::stoi(argv[1]);
        H = std::stoi(argv[2]);
        decisionVariableNum = std::stoi(argv[3]);
        neighborNum = std::stoi(argv[4]);
    } else if (argc >= 6) {
        std::cerr << "Usage: " << argv[0] << " [generationNum] [H] [decisionVariableNum] [neighborNum]" << std::endl;
        return 1;
    }
    std::shared_ptr<LZ1> problem = std::make_shared<LZ1>(decisionVariableNum);
    int objectiveNum = problem->objectiveNum;
    std::shared_ptr<BinomialCrossover> crossover = std::make_shared<BinomialCrossover>(1.0, 0.5);
    std::shared_ptr<Tchebycheff> decomposition = std::make_shared<Tchebycheff>();
    std::shared_ptr<PolynomialMutation> mutation =
        std::make_shared<PolynomialMutation>(1.0 / decisionVariableNum, 20.0, problem->variableBounds);
    std::shared_ptr<UniformRandomSampling> sampling =
        std::make_shared<UniformRandomSampling>(problem->variableBounds[0][0], problem->variableBounds[1][0]);
    std::shared_ptr<RandomSelection> selection = std::make_shared<RandomSelection>();

    Moead<double> moead(generationNum, decisionVariableNum, objectiveNum, neighborNum, H, crossover, decomposition, mutation,
                        problem, sampling, selection);
    moead.Run();

    std::ofstream ofs("out/data/result.txt");
    // Write the objective sets to the output file
    for (const auto& set : moead.objectiveSets) {
        for (const auto& value : set) {
            ofs << value << " ";
        }
        ofs << std::endl;
    }

    return 0;
}
