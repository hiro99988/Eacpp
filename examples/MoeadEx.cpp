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
#include "Mutations/PolynomialMutation.h"
#include "Problems/Problems.h"
#include "Reflections/Reflection.h"
#include "Repairs/RealRandomRepair.h"
#include "Samplings/RealRandomSampling.h"
#include "Selections/RandomSelection.h"

using namespace Eacpp;

int main(int argc, char* argv[]) {
    int generationNum = 500;
    int H = 299;
    int neighborNum = 21;
    std::string problemName = "zdt1";

    if (argc == 2) {
        generationNum = std::stoi(argv[1]);
    } else if (argc == 3) {
        generationNum = std::stoi(argv[1]);
        H = std::stoi(argv[2]);
    } else if (argc == 4) {
        generationNum = std::stoi(argv[1]);
        H = std::stoi(argv[2]);
        neighborNum = std::stoi(argv[3]);
    } else if (argc == 5) {
        generationNum = std::stoi(argv[1]);
        H = std::stoi(argv[2]);
        neighborNum = std::stoi(argv[3]);
        problemName = argv[4];
    }

    std::shared_ptr<IProblem<double>> problem = Reflection<IProblem<double>>::Create(problemName);

    auto crossover = std::make_shared<SimulatedBinaryCrossover>(0.9, problem->VariableBounds());
    auto decomposition = std::make_shared<Tchebycheff>();
    auto mutation =
        std::make_shared<PolynomialMutation>(1.0 / problem->DecisionVariablesNum(), 20.0, problem->VariableBounds());
    auto sampling = std::make_shared<RealRandomSampling>(problem->VariableBounds());
    auto repair = std::make_shared<RealRandomRepair>(problem);
    auto selection = std::make_shared<RandomSelection>();

    Moead<double> moead(generationNum, neighborNum, H, crossover, decomposition, mutation, problem, repair, sampling,
                        selection);

    auto start = std::chrono::system_clock::now();

    moead.Run();

    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Execution time: " << duration << "ms" << std::endl;

    std::filesystem::path objectiveFilePath = "out/data/tmp/zdt1.csv";
    std::ofstream objectiveFile(objectiveFilePath);
    for (const auto& objectives : moead.GetObjectivesList()) {
        for (size_t j = 0; j < objectives.size(); j++) {
            if (j == objectives.size() - 1) {
                objectiveFile << objectives[j];
            } else {
                objectiveFile << objectives[j] << ",";
            }
        }
        objectiveFile << std::endl;
    }

    return 0;
}
