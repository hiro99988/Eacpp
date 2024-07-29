#include <fstream>
#include <iostream>
#include <memory>

#include "Algorithms/Moead.h"
#include "Crossovers/OnePointCrossover.h"
#include "Decompositions/Tchebycheff.h"
#include "Mutations/PolynomialMutation.h"
#include "Problems/LZ1.h"
#include "Samplings/UniformRandomSampling.h"
#include "Selections/RandomSelection.h"

using namespace Eacpp;

int main() {
    int generationNum = 500;
    int decisionVariableNum = 10;
    int neighborNum = 20;
    int H = 299;
    std::shared_ptr<LZ1> problem = std::make_shared<LZ1>(decisionVariableNum);
    int objectiveNum = problem->objectiveNum;
    std::shared_ptr<OnePointCrossover<double>> crossover = std::make_shared<OnePointCrossover<double>>(1.0);
    std::shared_ptr<Tchebycheff> decomposition = std::make_shared<Tchebycheff>();
    std::shared_ptr<PolynomialMutation> mutation = std::make_shared<PolynomialMutation>(0.1, 20.0, problem->variableBounds);
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
