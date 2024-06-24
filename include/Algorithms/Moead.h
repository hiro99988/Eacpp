#ifndef Moead_H
#define Moead_H

#include "Crossovers/ICrossover.h"
#include "Decompositions/IDecomposition.h"
#include "Mutations/IMutation.h"
#include "Problems/IProblem.h"
#include "Samplings/ISampling.h"
#include "Selections/ISelection.h"

namespace Eacpp {

class Moead {
   public:
    int generationNum;
    int populationSize;
    int decisionVariablenum;
    int objectiveNum;
    int neighborNum;
    int H;
    ICrossover<double>& crossover;
    IDecomposition& decompositioin;
    IProblem& problem;
    ISampling<double>& sampling;
    ISelection<double>& selection;

    Moead(int generationNum, int populationSize, int decisionVariablenum, int objectiveNum, int neighborNum, int H,
          ICrossover<double>& crossover, IDecomposition& decompositioin, IProblem& problem, ISampling<double>& sampling,
          ISelection<double>& selection)
        : generationNum(generationNum),
          populationSize(populationSize),
          decisionVariablenum(decisionVariablenum),
          objectiveNum(objectiveNum),
          neighborNum(neighborNum),
          H(H),
          crossover(crossover),
          decompositioin(decompositioin),
          problem(problem),
          sampling(sampling),
          selection(selection) {}
};

}  // namespace Eacpp

#endif