#include "Problems/DTLZ.hpp"

#include <cmath>
#include <eigen3/Eigen/Core>
#include <numbers>
#include <tuple>
#include <vector>

#include "Individual.h"

namespace Eacpp {

int DTLZBase::DecisionVariablesNum() const {
    return _decisionVariablesNum;
}

int DTLZBase::ObjectivesNum() const {
    return _objectivesNum;
}

const std::vector<std::pair<double, double>>& DTLZBase::VariableBounds() const {
    return _variableBounds;
}

void DTLZBase::ComputeObjectiveSet(Individuald& individual) const {
    auto X_ = individual.solution.head(_objectivesNum - 1);
    auto XM = individual.solution.tail(individual.solution.size() - (_objectivesNum - 1));

    double g = G(XM);
    individual.objectives = Objectives(X_, g);
}

bool DTLZBase::IsFeasible(const Individuald& individual) const {
    return (individual.solution >= _variableBounds[0].first).all() && (individual.solution <= _variableBounds[0].second).all();
}

std::vector<bool> DTLZBase::EvaluateConstraints(const Individuald& individual) const {
    std::vector<bool> evaluation(individual.solution.size());
    for (int i = 0; i < individual.solution.size(); i++) {
        evaluation[i] =
            individual.solution(i) >= _variableBounds[0].first && individual.solution(i) <= _variableBounds[0].second;
    }

    return evaluation;
}

double DTLZBase::G13(const Eigen::ArrayXd& XM) const {
    double y = (XM - 0.5).square().sum() - (20.0 * std::numbers::pi * (XM - 0.5)).cos().sum();
    return 100.0 * (static_cast<double>(XM.size()) + y);
}

double DTLZBase::G245(const Eigen::ArrayXd& XM) const {
    return (XM - 0.5).square().sum();
}

Eigen::ArrayXd DTLZBase::Objectives234(const Eigen::ArrayXd& X_, double g, double alpha) const {
    Eigen::ArrayXd objectives(_objectivesNum);

    for (std::size_t i = 0; i < _objectivesNum; ++i) {
        double f = 1.0 + g;
        f *= (X_.head(X_.size() - i).pow(alpha) * std::numbers::pi / 2.0).cos().prod();
        if (i > 0) {
            f *= std::sin(std::pow(X_(X_.size() - i), alpha) * std::numbers::pi / 2.0);
        }

        objectives(i) = f;
    }

    return objectives;
}

Eigen::ArrayXd DTLZBase::Objectives56(const Eigen::ArrayXd& X_, double g) const {
    auto theta = theta56(X_, g);
    return Objectives234(theta, g);
}

Eigen::ArrayXd DTLZBase::theta56(const Eigen::ArrayXd& X_, double g) const {
    Eigen::ArrayXd theta = 1.0 / (2.0 * (1.0 + g)) * (1.0 + 2.0 * g * X_);
    theta(0) = X_(0);
    return theta;
}

Eigen::ArrayXd DTLZ1::Objectives(const Eigen::ArrayXd& X_, double g) const {
    Eigen::ArrayXd objectives(ObjectivesNum());

    for (std::size_t i = 0; i < objectives.size(); ++i) {
        double f = 0.5 * (1.0 + g);
        f *= X_.head(X_.size() - i).prod();
        if (i > 0) {
            f *= 1.0 - X_(X_.size() - i);
        }

        objectives(i) = f;
    }

    return objectives;
}

double DTLZ6::G(const Eigen::ArrayXd& XM) const {
    return XM.pow(0.1).sum();
}

double DTLZ7::G(const Eigen::ArrayXd& XM) const {
    return 1.0 + 9.0 / static_cast<double>(XM.size()) * XM.sum();
}

Eigen::ArrayXd DTLZ7::Objectives(const Eigen::ArrayXd& X_, double g) const {
    Eigen::ArrayXd objectives(ObjectivesNum());

    for (std::size_t i = 0; i < objectives.size() - 1; ++i) {
        objectives(i) = X_(i);
    }

    double h = H(objectives.head(objectives.size() - 1), g);
    objectives(objectives.size() - 1) = (1.0 + g) * h;

    return objectives;
}

double DTLZ7::H(const Eigen::ArrayXd& F_, double g) const {
    return static_cast<double>(ObjectivesNum()) - ((F_ / (1.0 + g)) * (1.0 + (3.0 * std::numbers::pi * F_).sin())).sum();
}

}  // namespace Eacpp