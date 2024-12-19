#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <set>
#include <vector>

#include "Graph/NetworkTopologySearch.hpp"
#include "Graph/SimpleGraph.hpp"

using namespace Eacpp;

int main() {
    std::ifstream file("data/inputs/graphParameter.json");
    nlohmann::json json = nlohmann::json::parse(file);

    int objectivesNum = json["objectivesNum"];
    int neighborhoodSize = json["neighborhoodSize"];
    int divisionsNumOfWeightVector = json["divisionsNumOfWeightVector"];
    int nodesNum = json["nodesNum"];
    int degree = json["degree"];
    int repeats = json["repeats"];
    double initialTemperature = json["initialTemperature"];
    double minTemperature = json["minTemperature"];
    double coolingRate = json["coolingRate"];
    double weightOfAsplNeighborsInObjective = json["weightOfAsplNeighborsInObjective"];
    double weightOfAsplExtremesInObjective = json["weightOfAsplExtremesInObjective"];
    bool isOutput = json["isOutput"];

    NetworkTopologySearch search(objectivesNum, neighborhoodSize, divisionsNumOfWeightVector, nodesNum, degree,
                                 weightOfAsplNeighborsInObjective, weightOfAsplExtremesInObjective, isOutput);

    search.Run(repeats, initialTemperature, minTemperature, coolingRate);

    return 0;
}