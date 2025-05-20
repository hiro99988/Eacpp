#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Rng/IRng.h"
#include "Rng/Rng.h"

namespace Eacpp {

/// @brief 単一目的最適化問題のインターフェース
/// @tparam SolutionType 解の型
template <typename SolutionType>
class SingleObjectiveProblem {
   public:
    /// @brief 目的関数を計算する純粋仮想関数
    /// @param solution 解
    /// @return 解の目的値
    virtual double ComputeObjective(const SolutionType& solution) = 0;
};

/// @brief 近傍解生成関数のインターフェース
/// @tparam SolutionType 解の型
template <typename SolutionType>
class NeighborGenerator {
   public:
    /// @brief 近傍解生成関数の純粋仮想関数
    /// @param currentSolution 現在の解
    /// @return 新しい近傍解
    virtual SolutionType Generate(const SolutionType& currentSolution) = 0;
};

template <typename SolutionType>
class SimulatedAnnealing {
   public:
    // ///@brief
    // SAの現在の状態を保持する構造体。進捗コールバック関数に渡されます。 struct
    // SAState {
    //     double temperature;           ///< 現在の温度
    //     int iterationsAtCurrentTemp;  ///< 現在の温度でのイテレーション回数
    //     long long totalIterations;    ///< 総イテレーション回数
    //     const SolutionType& currentSolution;  ///< 現在の解への参照
    //     double currentEnergy;                 ///< 現在の解のエネルギー
    //     const SolutionType&
    //         bestSolutionSoFar;   ///< これまでに見つかった最良解への参照
    //     double bestEnergySoFar;  ///< これまでに見つかった最良解のエネルギー
    //     int iterationsSinceLastImprovement;  ///<
    //     最良解が最後に改善されてからのイテレーション回数
    ///@brief SAの現在の状態を保持する構造体。進捗コールバック関数に渡される
    struct SAState {
        double temperature;           /// 現在の温度
        int iterationsAtCurrentTemp;  /// 現在の温度でのイテレーション回数
        long long totalIterations;    /// 総イテレーション回数
        const SolutionType& currentSolution;  /// 現在の解への参照
        double currentObjective;              /// 現在の解の目的値
        const SolutionType& bestSoFar;  /// これまでに見つかった最解への参照
        double bestSoFarObjective;      /// これまでに見つかった最良解の目的値
        int iterationsSinceLastImprovement;  /// 最良解が最後に改善されてからのイテレーション回数

        SAState(double temp, int iterAtTemp, long long totalIter,
                const SolutionType& currentSol, double currentObj,
                const SolutionType& bestSol, double bestObj, int stagnantIter)
            : temperature(temp),
              iterationsAtCurrentTemp(iterAtTemp),
              totalIterations(totalIter),
              currentSolution(currentSol),
              currentObjective(currentObj),
              bestSoFar(bestSol),
              bestSoFarObjective(bestObj),
              iterationsSinceLastImprovement(stagnantIter) {}
    };

    /// @brief 進捗通知コールバック関数の型エイリアス。
    /// SAStateオブジェクトを定数参照で受け取ります。
    using ProgressCallback = std::function<void(const SAState&)>;

    /// @brief SimulatedAnnealing クラスのコンストラクタ。
    /// @param initialSolution 初期解
    /// @param singleObjectiveProblem 解の目的値を評価する関数
    /// @param neighborGen 近傍解を生成する関数
    /// @param initialTemperature 初期温度
    /// @param coolingRate 冷却率
    /// @param minTemperature 最低温度
    /// @param maxIterationsPerTemperature
    /// 各温度で試行する最大イテレーション回数
    /// @param maxTotalIterations
    /// 総イテレーション回数の上限。0以下の場合は無制限。
    /// @param maxStagnantIterations
    /// 最良解が更新されない場合の許容最大イテレーション回数。0以下の場合は無制限。
    /// @param verbose 進捗通知を表示するかどうか
    /// @param seed
    /// 乱数生成器のシード。デフォルトでは現在時刻に基づいた値を使用します。
    SimulatedAnnealing(
        SolutionType initialSolution,
        std::unique_ptr<SingleObjectiveProblem<SolutionType>>
            singleObjectiveProblem,
        std::unique_ptr<NeighborGenerator<SolutionType>> neighborGen,
        double initialTemperature, double coolingRate, double minTemperature,
        int maxIterationsPerTemperature, long long maxTotalIterations = 0,
        int maxStagnantIterations = 0, bool verbose = false,
        std::uint_fast32_t seed = std::random_device()(),
        ProgressCallback callback = nullptr)
        : _originalInitialSolution(std::move(initialSolution)),
          _problem(std::move(singleObjectiveProblem)),
          _neighborGenerator(std::move(neighborGen)),
          _initialTemperature(initialTemperature),
          _temperature(initialTemperature),
          _coolingRate(coolingRate),
          _minTemperature(minTemperature),
          _maxIterationsPerTemp(maxIterationsPerTemperature),
          _maxTotalIterations(maxTotalIterations),
          _maxStagnantIterations(maxStagnantIterations),
          _verbose(verbose),
          _rng(seed),
          _progressCallback(callback) {
        // パラメータの検証
        if (_problem == nullptr) {
            throw std::invalid_argument(
                "目的関数はnullptrであってはなりません。");
        }
        if (_neighborGenerator == nullptr) {
            throw std::invalid_argument(
                "近傍解生成関数はnullptrであってはなりません。");
        }
        if (_initialTemperature <= 0) {
            throw std::invalid_argument("初期温度は正である必要があります。");
        }
        if (_coolingRate <= 0 || _coolingRate >= 1.0) {
            throw std::invalid_argument(
                "冷却率は0より大きく1未満である必要があります。");
        }
        if (_minTemperature < 0 || _minTemperature >= _initialTemperature) {
            throw std::invalid_argument(
                "最低温度は0以上かつ初期温度未満である必要があります。");
        }
        if (_maxIterationsPerTemp <= 0) {
            throw std::invalid_argument(
                "各温度での最大イテレーション回数は正である必要があります"
                "。");
        }
        Reset();  // 状態を初期化
        SetDefaultProgressCallback();
    }

    /// @brief
    /// SAアルゴリズムの内部状態を初期状態（コンストラクタで指定された初期解と温度）にリセットします。
    void Reset() {
        _currentSolution = _originalInitialSolution;
        _bestSoFar = _originalInitialSolution;
        _currentObjective = _problem->ComputeObjective(_currentSolution);
        _bestSoFarObjective = _currentObjective;
        _temperature = _initialTemperature;
        _totalIterations = 0;
        _iterationsSinceLastImprovement = 0;
    }

    /// @brief SAアルゴリズムの内部状態を新しい初期解でリセットします。
    /// @param newInitialSolution
    /// 新しい初期解。ムーブ可能な型であればムーブされます。
    void Reset(SolutionType newInitialSolution) {
        _originalInitialSolution = std::move(newInitialSolution);
        Reset();
    }

    /// @brief Simulated Annealing アルゴリズムを実行します
    /// @return 見つかった最良解。SolutionTypeのコピーが返されます
    SolutionType Run() {
        bool stop = false;
        _iterationsSinceLastImprovement = 0;

        if (_verbose) {
            _progressCallback(SAState(_temperature, 0, _totalIterations,
                                      _currentSolution, _currentObjective,
                                      _bestSoFar, _bestSoFarObjective,
                                      _iterationsSinceLastImprovement));
        }

        while (!stop) {
            for (int i = 0; i < _maxIterationsPerTemp && !stop; ++i) {
                // 1. 近傍解を生成
                SolutionType neighborSolution =
                    _neighborGenerator->Generate(_currentSolution);
                // 2. 近傍解の目的値を計算
                double neighborObjective =
                    _problem->ComputeObjective(neighborSolution);

                // 3. 目的値の変化量を計算
                double deltaObjective = neighborObjective - _currentObjective;
                bool accepted = false;

                if (deltaObjective <
                    0) {  // 新しい解の方が良い場合、無条件に受理
                    accepted = true;
                } else {  // 新しい解の方が悪い場合、確率で受理
                          // (Metropolis基準)
                    if (_temperature > 1e-9 &&
                        _rng.Random() <
                            std::exp(-deltaObjective / _temperature)) {
                        accepted = true;
                    }
                }

                if (accepted) {
                    _currentSolution = std::move(
                        neighborSolution);  // 近傍解をムーブ(またはコピー)で現在の解に
                    _currentObjective = neighborObjective;

                    if (_currentObjective < _bestSoFarObjective) {
                        _bestSoFar = _currentSolution;  // 最良解を更新 (コピー)
                        _bestSoFarObjective = _currentObjective;
                        _iterationsSinceLastImprovement = 0;
                    }
                }
                _totalIterations++;
                _iterationsSinceLastImprovement++;

                // 終了条件のチェック
                if (_maxTotalIterations > 0 &&
                    _totalIterations >= _maxTotalIterations) {
                    stop = true;
                    if (_verbose) {
                        std::cout << "最大イテレーション回数に達しました。"
                                  << std::endl;
                    }
                }
                if (_maxStagnantIterations > 0 &&
                    _iterationsSinceLastImprovement >= _maxStagnantIterations) {
                    stop = true;
                    if (_verbose) {
                        std::cout << "最良解が更新されないイテレーション回数が"
                                  << _maxStagnantIterations << "に達しました。"
                                  << std::endl;
                    }
                }

                // 進捗コールバックの呼び出し
                if (_progressCallback) {
                    _progressCallback(SAState(
                        _temperature, i + 1, _totalIterations, _currentSolution,
                        _currentObjective, _bestSoFar, _bestSoFarObjective,
                        _iterationsSinceLastImprovement));
                }
            }
            // 4. 温度を更新 (冷却)
            _temperature *= _coolingRate;
            _temperature = std::max(_temperature, _minTemperature);
        }
        return _bestSoFar;
    }

    /// @brief 見つかった最良解のエネルギーを取得する
    /// @return 最良解のエネルギー。
    double GetBestSoFarObjective() const {
        return _bestSoFarObjective;
    }

    /// @brief 見つかった最良解を取得する
    /// @return 最良解への定数参照。
    const SolutionType& GetBestSoFar() const {
        return _bestSoFar;
    }

    /// @brief 現在の総イテレーション回数を取得する
    /// @return 総イテレーション回数。
    long long GetTotalIterations() const {
        return _totalIterations;
    }

    /// @brief 現在の温度を取得する
    /// @return 現在の温度。
    double GetCurrentTemperature() const {
        return _temperature;
    }

   private:
    SolutionType _originalInitialSolution;  // リセット用に保持する初期解
    SolutionType _currentSolution;          // 現在の解
    SolutionType _bestSoFar;                // これまでに見つかった最良解
    double _currentObjective;               // 現在の解のエネルギー
    double _bestSoFarObjective;             // 最良解のエネルギー

    std::unique_ptr<SingleObjectiveProblem<SolutionType>> _problem;  // 目的関数
    std::unique_ptr<NeighborGenerator<SolutionType>>
        _neighborGenerator;              // 近傍解生成関数
    ProgressCallback _progressCallback;  // 進捗通知コールバック

    double _initialTemperature;  // 初期温度
    double _temperature;         // 現在の温度
    double _coolingRate;         // 冷却率
    double _minTemperature;      // 最低温度
    int _maxIterationsPerTemp;   // 各温度での最大イテレーション回数

    long long _maxTotalIterations;  // 総イテレーション回数の上限
    int _maxStagnantIterations;     // 停滞許容イテレーション回数

    long long _totalIterations;  // 実行された総イテレーション回数
    int _iterationsSinceLastImprovement;  // 最良解が更新されずに経過したイテレーション回数

    bool _verbose;  // 進捗通知を表示するかどうか

    Rng _rng;  // 乱数生成器

   private:
    /// @brief デフォルトの進捗通知コールバック関数
    /// @param state SAの状態
    void DefaultProgressCallback(const SAState& state) {
        std::cout << "iters: " << state.totalIterations << ", stagnant iters: "
                  << state.iterationsSinceLastImprovement
                  << ", temp: " << state.temperature
                  << ", iters at temp: " << state.iterationsAtCurrentTemp
                  << ", current obj: " << state.currentObjective
                  << ", best obj: " << state.bestSoFarObjective << std::endl;
    }

    /// @brief デフォルトの進捗通知コールバック関数を設定する
    /// @param callback コールバック関数
    void SetDefaultProgressCallback() {
        if (!_progressCallback) {
            _progressCallback = [this](const SAState& state) {
                DefaultProgressCallback(state);
            };
        }
    }
};

template <typename T>
using SA = SimulatedAnnealing<T>;

}  // namespace Eacpp