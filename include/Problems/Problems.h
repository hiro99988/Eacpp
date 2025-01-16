#pragma once

#include "Problems/DTLZ.hpp"
#include "Problems/ZDT1.h"
#include "Problems/ZDT2.h"
#include "Problems/ZDT3.h"
#include "Problems/ZDT4.h"
#include "Problems/ZDT6.h"

namespace Eacpp {

inline std::unique_ptr<IProblem<double>> CreateProblem(const std::string& name,
                                                       int decisionVariablesNum,
                                                       int objectivesNum) {
    std::string lowerName = name;
    std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(),
                   ::tolower);

    if (lowerName == "dtlz1") {
        return std::make_unique<DTLZ1>(decisionVariablesNum, objectivesNum);
    } else if (lowerName == "dtlz2") {
        return std::make_unique<DTLZ2>(decisionVariablesNum, objectivesNum);
    } else if (lowerName == "dtlz3") {
        return std::make_unique<DTLZ3>(decisionVariablesNum, objectivesNum);
    } else if (lowerName == "dtlz4") {
        return std::make_unique<DTLZ4>(decisionVariablesNum, objectivesNum);
    } else if (lowerName == "dtlz5") {
        return std::make_unique<DTLZ5>(decisionVariablesNum, objectivesNum);
    } else if (lowerName == "dtlz6") {
        return std::make_unique<DTLZ6>(decisionVariablesNum, objectivesNum);
    } else if (lowerName == "dtlz7") {
        return std::make_unique<DTLZ7>(decisionVariablesNum, objectivesNum);
    } else if (lowerName == "zdt1") {
        return std::make_unique<ZDT1>(decisionVariablesNum);
    } else if (lowerName == "zdt2") {
        return std::make_unique<ZDT2>(decisionVariablesNum);
    } else if (lowerName == "zdt3") {
        return std::make_unique<ZDT3>(decisionVariablesNum);
    } else if (lowerName == "zdt4") {
        return std::make_unique<ZDT4>(decisionVariablesNum);
    } else if (lowerName == "zdt6") {
        return std::make_unique<ZDT6>(decisionVariablesNum);
    } else {
        throw std::invalid_argument("Unknown problem name: " + name);
    }
}

}  // namespace Eacpp