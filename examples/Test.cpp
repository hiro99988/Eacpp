#include <pybind11/embed.h>

#include <Eigen/Core>

#include "Individual.h"
#include "Problems/Hpa.hpp"

using namespace Eacpp;
namespace py = pybind11;

int main(int argc, char* argv[]) {
    // Pythonインタプリタを初期化
    py::scoped_interpreter guard{};
    // Pythonモジュールをインポート
    py::module module = py::module::import("extern.hpa.hpa.problem");

    // コマンドライン引数から問題名を取得
    std::string problemName = "HPA201";
    int level = 0;
    if (argc == 2) {
        problemName = argv[1];
    } else if (argc == 3) {
        problemName = argv[1];
        level = std::stoi(argv[2]);
    }

    Hpa hpa(module, problemName.c_str(), 4, level);

    Eigen::ArrayXd x(hpa.DecisionVariablesNum());
    x.setLinSpaced(x.size(), 0.0, 1.0);
    Individuald individual(x);
    hpa.ComputeObjectiveSet(individual);

    std::cout << "Decision Variables: " << individual.solution.transpose()
              << std::endl;
    std::cout << "Objectives: " << individual.objectives.transpose()
              << std::endl;

    return 0;
}