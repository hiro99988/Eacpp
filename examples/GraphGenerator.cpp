#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <set>
#include <vector>

#include "Graph/NetworkTopologySearch.hpp"
#include "Graph/SimpleGraph.hpp"

using namespace Eacpp;

int main() {
    int objectivesNum;
    int neighborhoodSize;
    int divisionsNumOfWeightVector;
    int nodesNum;
    int degree;
    int idealPathLengthBetweenNeighbors;
    int idealPathLengthBetweenExtremesAndAnyNode;
    int repeats;
    double initialTemperature;
    double minTemperature;
    double coolingRate;

    std::ifstream file("data/inputs/graphParameter.json");
    nlohmann::json json = nlohmann::json::parse(file);

    objectivesNum = json["objectivesNum"];
    neighborhoodSize = json["neighborhoodSize"];
    divisionsNumOfWeightVector = json["divisionsNumOfWeightVector"];
    nodesNum = json["nodesNum"];
    degree = json["degree"];
    idealPathLengthBetweenNeighbors = json["idealPathLengthBetweenNeighbors"];
    idealPathLengthBetweenExtremesAndAnyNode = json["idealPathLengthBetweenExtremesAndAnyNode"];
    repeats = json["repeats"];
    initialTemperature = json["initialTemperature"];
    minTemperature = json["minTemperature"];
    coolingRate = json["coolingRate"];

    NetworkTopologySearch search(objectivesNum, neighborhoodSize, divisionsNumOfWeightVector, nodesNum, degree,
                                 idealPathLengthBetweenNeighbors, idealPathLengthBetweenExtremesAndAnyNode);

    search.Run(repeats, initialTemperature, minTemperature, coolingRate);

    return 0;
}