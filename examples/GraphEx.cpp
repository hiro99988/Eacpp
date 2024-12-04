#include <iostream>
#include <set>
#include <vector>

#include "Graph/NetworkTopologySearch.hpp"
#include "Graph/SimpleGraph.hpp"

using namespace Eacpp;

int main() {
    int objectivesNum = 2;
    int neighborhoodSize = 21;
    int divisionsNumOfWeightVector = 299;
    int nodesNum = 50;
    int degree = 4;
    int idealPathLengthBetweenNeighbors = 2;
    int idealPathLengthBetweenExtremesAndAnyNode = 4;
    int repeats = 1'000'000;
    double initialTemperature = 1000;
    double minTemperature = 0.001;
    double coolingRate = 0.995;

    NetworkTopologySearch search(objectivesNum, neighborhoodSize, divisionsNumOfWeightVector, nodesNum, degree,
                                 idealPathLengthBetweenNeighbors, idealPathLengthBetweenExtremesAndAnyNode);

    search.Run(repeats, initialTemperature, minTemperature, coolingRate);

    return 0;
}