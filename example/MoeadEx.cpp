#include <iostream>

#include "Algorithms/Moead.h"
#include "Crossovers/OnePointCrossover.h"
#include "Decompositions/Tchebycheff.h"
#include "Mutations/PolynomialMutation.h"
#include "Problems/LZ1.h"
#include "Samplings/UniformRandomSampling.h"
#include "Selections/RandomSelection.h"

using namespace Eacpp;

int main() {
    int generationNum = 100;
    int populationSize = 300;
    int decisionVariableNum = 10;
    int neighborNum = 20;
    int H = 299;
    LZ1 problem(decisionVariableNum);
    int objectiveNum = problem.objectiveNum;
    OnePointCrossover<double> crossover(1.0);
    Tchebycheff decomposition;
    PolynomialMutation mutation(0.1, 20.0, problem.variableBounds);
    UniformRandomSampling sampling(problem.variableBounds[0][0], problem.variableBounds[1][0]);
    RandomSelection<double> selection;

    Moead<double> moead(generationNum, populationSize, decisionVariableNum, objectiveNum, neighborNum, H, &crossover,
                        &decomposition, &mutation, &problem, &sampling, &selection);

    return 0;
}
