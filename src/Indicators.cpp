#include "Indicators.hpp"

#include <algorithm>
#include <eigen3/Eigen/Core>
#include <iostream>
#include <limits>
#include <vector>

#include "Utils/EigenUtils.h"

namespace Eacpp {

IGD::IGD(const std::vector<std::vector<double>>& paretoFront) {
    _paretoFront.reserve(paretoFront.size());
    for (const auto& point : paretoFront) {
        _paretoFront.push_back(
            Eigen::Map<const Eigen::ArrayXd>(point.data(), point.size()));
    }
}

double IGD::Calculate(const std::vector<Eigen::ArrayXd>& objectives) {
    double sum = 0.0;
    for (const auto& paretoPoint : _paretoFront) {
        double minDistance = std::numeric_limits<double>::max();
        for (const auto& objective : objectives) {
            double distance =
                CalculateEuclideanDistance(objective, paretoPoint);
            minDistance = std::min(minDistance, distance);
        }

        sum += minDistance;
    }

    return sum / static_cast<double>(_paretoFront.size());
}

double IGD::Calculate(const std::vector<std::vector<double>>& objectives) {
    std::vector<Eigen::ArrayXd> objectivesEigen;
    for (const auto& point : objectives) {
        objectivesEigen.push_back(
            Eigen::Map<const Eigen::ArrayXd>(point.data(), point.size()));
    }

    return Calculate(objectivesEigen);
}

IGDPlus::IGDPlus(const std::vector<std::vector<double>>& paretoFront) {
    _paretoFront.reserve(paretoFront.size());
    for (const auto& point : paretoFront) {
        _paretoFront.push_back(
            Eigen::Map<const Eigen::ArrayXd>(point.data(), point.size()));
    }
}

double IGDPlus::Calculate(const std::vector<Eigen::ArrayXd>& objectives) {
    double sum = 0.0;
    for (const auto& pf : _paretoFront) {
        double minDistance = std::numeric_limits<double>::max();
        for (const auto& objective : objectives) {
            double distance = CalculateModifiedDistance(objective, pf);
            minDistance = std::min(minDistance, distance);
        }
        sum += minDistance;
    }
    return sum / _paretoFront.size();
}

double IGDPlus::Calculate(const std::vector<std::vector<double>>& objectives) {
    std::vector<Eigen::ArrayXd> objectivesEigen;
    for (const auto& point : objectives) {
        objectivesEigen.push_back(
            Eigen::Map<const Eigen::ArrayXd>(point.data(), point.size()));
    }

    return Calculate(objectivesEigen);
}

}  // namespace Eacpp