#include "Problems/Hpa.hpp"

#include <pybind11/eigen.h>
#include <pybind11/embed.h>

#include <Eigen/Core>

#include "Individual.h"

namespace Eacpp {

Hpa::Hpa(const pybind11::module& module, const char* problemName, int n_div,
         int level, bool normalize) {
    // 問題インスタンスを生成
    _hpa = module.attr(problemName)(n_div, level, normalize);
    // nx: 決定変数の数
    _decisionVariablesNum = _hpa.attr("nx").cast<int>();
    // nf: 目的関数の数
    _objectivesNum = _hpa.attr("nf").cast<int>();
}

int Hpa::DecisionVariablesNum() const {
    return _decisionVariablesNum;
}

int Hpa::ObjectivesNum() const {
    return _objectivesNum;
}

const std::vector<std::pair<double, double>>& Hpa::VariableBounds() const {
    return _variableBounds;
}

void Hpa::ComputeObjectiveSet(Individuald& individual) const {
    // Eigen::ArrayX を py::array に変換
    pybind11::array_t<double> x = pybind11::cast(individual.solution);
    // Python関数を呼び出す
    pybind11::object result = _hpa.attr("__call__")(x);
    // 返り値を1次元配列として取得
    pybind11::array arr = result.cast<pybind11::array>();
    // 返り値をEigen::ArrayXdに変換
    pybind11::buffer_info buf = arr.request();
    double* ptr = static_cast<double*>(buf.ptr);
    size_t size = buf.shape[0];
    Eigen::ArrayXd objectives = Eigen::Map<Eigen::ArrayXd>(ptr, size);

    individual.objectives = objectives;
}

bool Hpa::IsFeasible(const Individuald& individual) const {
    return (individual.solution >= _variableBounds[0].first).all() &&
           (individual.solution <= _variableBounds[0].second).all();
}

std::vector<bool> Hpa::EvaluateConstraints(
    const Individuald& individual) const {
    std::vector<bool> evaluation(individual.solution.size());
    for (int i = 0; i < individual.solution.size(); i++) {
        evaluation[i] = individual.solution(i) >= _variableBounds[0].first &&
                        individual.solution(i) <= _variableBounds[0].second;
    }

    return evaluation;
}

}  // namespace Eacpp