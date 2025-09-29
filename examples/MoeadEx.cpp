#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
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
#include "Utils/FileUtils.h"

using namespace eacpp;

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
    }

    std::shared_ptr<IProblem<double>> problem =
        Reflection<IProblem<double>>::Create("zdt1");

    auto crossover = std::make_shared<SimulatedBinaryCrossover>(
        0.9, problem->VariableBounds());
    auto decomposition = std::make_shared<Tchebycheff>();
    auto mutation = std::make_shared<PolynomialMutation>(
        1.0 / problem->DecisionVariablesNum(), 20.0, problem->VariableBounds());
    auto sampling =
        std::make_shared<RealRandomSampling>(problem->VariableBounds());
    auto repair = std::make_shared<RealRandomRepair>(problem);
    auto selection = std::make_shared<RandomSelection>();

    Moead<double> moead(generationNum, neighborNum, H, crossover, decomposition,
                        mutation, problem, repair, sampling, selection);

    auto start = std::chrono::high_resolution_clock::now();
    moead.Run();
    auto end = std::chrono::high_resolution_clock::now();

    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    std::cout << "Elapsed time: " << duration << "ms" << std::endl;

    std::filesystem::path paretoFrontFilePath =
        "data/ground_truth/pareto_fronts/ZDT1.csv";
    auto paretoFrontFile = OpenInputFile(paretoFrontFilePath);
    auto paretoFront = ReadCsv<double>(paretoFrontFile, true, true);
    IGDPlus indicator(paretoFront);
    double igdPlusValue = indicator.Calculate(moead.GetObjectivesList());
    std::cout << "IGD+: " << igdPlusValue << std::endl;

    return 0;
}
