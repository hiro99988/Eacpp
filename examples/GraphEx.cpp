#include "Graph/NetworkTopologySearch.hpp"
#include "Graph/SimpleGraph.hpp"

using namespace Eacpp;

int main() {
    int objectivesNum = 2;
    int neighborhoodSize = 21;
    int divisionsNumOfWeightVector = 300;
    int nodesNum = 50;
    int idealPathLengthBetweenNeighbors = 2;
    int idealPathLengthBetweenExtremesAndAnyNode = 3;
    int repeats = 100000;
    double initialTemperature = 1000;
    double minTemperature = 0.01;
    double coolingRate = 0.9993;

    NetworkTopologySearch search(objectivesNum, neighborhoodSize, divisionsNumOfWeightVector, nodesNum,
                                 idealPathLengthBetweenNeighbors, idealPathLengthBetweenExtremesAndAnyNode, repeats,
                                 initialTemperature, minTemperature, coolingRate);

    search.Initialize();
    search.Search();

    return 0;
}