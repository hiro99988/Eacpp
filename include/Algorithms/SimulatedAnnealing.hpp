#pragma once

#include <cstdint>
#include <eigen3/Eigen/Core>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "Rng/IRng.h"
#include "Rng/Rng.h"

namespace Eacpp {

// TODO: Run関数で初期個体を引数で受け取ってSAを動かすように
// TODO; Evaluate関数はSA外で定義した関数 or クラスを受け取る
template <typename T>
class SimulatedAnnealing {
   public:
    SimulatedAnnealing(int repeats, double initialTemperature,
                       double minTemperature, double coolingRate)
        : _repeats(repeats),
          _initialTemperature(initialTemperature),
          _minTemperature(minTemperature),
          _coolingRate(coolingRate),
          _rng(std::make_unique<Rng>()) {}

    SimulatedAnnealing(int repeats, double initialTemperature,
                       double minTemperature, double coolingRate,
                       std::uint_fast32_t seed)
        : _repeats(repeats),
          _initialTemperature(initialTemperature),
          _minTemperature(minTemperature),
          _coolingRate(coolingRate),
          _rng(std::make_unique<Rng>(seed)) {}

    SimulatedAnnealing(int repeats, double initialTemperature,
                       double minTemperature, double coolingRate,
                       std::unique_ptr<IRng> rng)
        : _repeats(repeats),
          _initialTemperature(initialTemperature),
          _minTemperature(minTemperature),
          _coolingRate(coolingRate),
          _rng(std::move(rng)) {}

    void Run() {
        Initialize();
        Search();
    }
    void Initialize() {}
    void Search();
    double BestSoFarObjective() const {
        return _bestSoFarObjective;
    }
    T BestSoFar() const {
        return _bestSoFarGraph;
    }
    double Evaluate(const T& graph) const;

   private:
    int _repeats;
    double _initialTemperature;
    double _minTemperature;
    double _coolingRate;
    std::unique_ptr<IRng> _rng;
    double _bestSoFarObjective;
    T _bestSoFarGraph;

    bool AcceptanceCriterion(double newObjective, double oldObjective,
                             double temperature) const {
        double delta = newObjective - oldObjective;
        return delta <= 0 || _rng->Random() < std::exp(-delta / (temperature));
    }

    void UpdateTemperature(double& temperature, double minTemperature,
                           double coolingRate) const {
        temperature *= coolingRate;
        temperature = std::max(temperature, minTemperature);
    }
};

template <typename T>
using SA = SimulatedAnnealing<T>;

}  // namespace Eacpp