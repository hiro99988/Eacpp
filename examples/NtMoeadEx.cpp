#include <mpi.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <tuple>
#include <vector>

#include "Algorithms/NtMoead.h"
#include "Crossovers/SimulatedBinaryCrossover.h"
#include "Decompositions/Tchebycheff.h"
#include "Mutations/PolynomialMutation.h"
#include "Problems/ZDT1.h"
#include "Reflections/Reflection.h"
#include "Repairs/RealRandomRepair.h"
#include "Samplings/RealRandomSampling.h"
#include "Selections/RandomSelection.h"

using namespace Eacpp;

int main() {
    auto problem = std::make_shared<ZDT1>();
    auto crossover = std::make_shared<SimulatedBinaryCrossover>(0.9, problem->VariableBounds());
    auto decomposition = std::make_shared<Tchebycheff>();
    auto mutation = std::make_shared<PolynomialMutation>(1.0 / problem->DecisionVariablesNum(), problem->VariableBounds());
    auto repair = std::make_shared<RealRandomRepair>(problem);
    auto sampling = std::make_shared<RealRandomSampling>(problem->VariableBounds());
    auto selection = std::make_shared<RandomSelection>();

    auto ntMoead = NtMoead<double>(500, 21, 59, 1, crossover, decomposition, mutation, problem, repair, sampling, selection);

    ntMoead.Initialize();

    return 0;
}