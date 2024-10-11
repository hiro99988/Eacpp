#pragma once

#include <eigen3/Eigen/Core>

namespace Eacpp {

constexpr double EpsilonD = 1.0e-13;
constexpr float EpsilonF = 1.0e-6f;

template <typename T>
bool AreEqual(const Eigen::ArrayX<T>& lhs, const Eigen::ArrayX<T>& rhs) {
    return lhs.size() == rhs.size() && (lhs == rhs).all();
}

template <typename T>
bool AreNotEqual(const Eigen::ArrayX<T>& lhs, const Eigen::ArrayX<T>& rhs) {
    return lhs.size() != rhs.size() || (lhs != rhs).any();
}

inline bool AreClose(const Eigen::ArrayXd& lhs, const Eigen::ArrayXd& rhs, double epsilon = EpsilonD) {
    return lhs.size() == rhs.size() && (lhs - rhs).isZero(epsilon);
}

inline bool AreClose(const Eigen::ArrayXf& lhs, const Eigen::ArrayXf& rhs, float epsilon = EpsilonF) {
    return lhs.size() == rhs.size() && (lhs - rhs).isZero(epsilon);
}

}  // namespace Eacpp