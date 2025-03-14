#pragma once

#include <cassert>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace Eacpp {

/// @brief 2つのvectorのサイズが一致しているかを検証する関数
/// @tparam T vectorの要素の型
/// @tparam Alloc vectorのアロケーター型
/// @param lhs 入力vector
/// @param rhs 入力vector
/// @param context エラーメッセージに表示するコンテキスト
/// @throw std::invalid_argument vectorのサイズが一致しない場合
template <typename T, typename Alloc>
inline void ValidateSameSize(const std::vector<T, Alloc>& lhs,
                             const std::vector<T, Alloc>& rhs,
                             const char* context) {
    if (lhs.size() != rhs.size()) {
        throw std::invalid_argument(std::string(context) +
                                    ": vector sizes do not match");
    }
}

/// @brief 2つのvectorの各要素の加算を行い，新しいvectorを返す関数
/// @tparam T vectorの要素の型
/// @tparam Alloc vectorのアロケーター型
/// @param lhs 入力vector
/// @param rhs 入力vector
/// @return 各要素を加算した結果のvector
template <typename T, typename Alloc>
inline std::vector<T, Alloc> Add(const std::vector<T, Alloc>& lhs,
                                 const std::vector<T, Alloc>& rhs) {
    ValidateSameSize(lhs, rhs, __func__);

    std::vector<T, Alloc> result(lhs.size());
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        result[i] = lhs[i] + rhs[i];
    }

    return result;
}

/// @brief 2つのvectorの各要素の減算を行い，新しいvectorを返す関数
/// @tparam T vectorの要素の型
/// @tparam Alloc vectorのアロケーター型
/// @param lhs 入力vector
/// @param rhs 入力vector
/// @return 各要素を減算した結果のvector
template <typename T, typename Alloc>
inline std::vector<T, Alloc> Subtract(const std::vector<T, Alloc>& lhs,
                                      const std::vector<T, Alloc>& rhs) {
    ValidateSameSize(lhs, rhs, __func__);

    std::vector<T, Alloc> result(lhs.size());
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        result[i] = lhs[i] - rhs[i];
    }

    return result;
}

/// @brief 2つのvectorの各要素の乗算を行い，新しいvectorを返す関数
/// @tparam T vectorの要素の型
/// @tparam Alloc vectorのアロケーター型
/// @param lhs 入力vector
/// @param rhs 入力vector
/// @return 各要素を掛け合わせた結果のvector
template <typename T, typename Alloc>
inline std::vector<T, Alloc> Multiply(const std::vector<T, Alloc>& lhs,
                                      const std::vector<T, Alloc>& rhs) {
    ValidateSameSize(lhs, rhs, __func__);

    std::vector<T, Alloc> result(lhs.size());
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        result[i] = lhs[i] * rhs[i];
    }

    return result;
}

/// @brief 2つのvectorの各要素の除算を行い，新しいvectorを返す関数
/// @tparam T vectorの要素の型
/// @tparam Alloc vectorのアロケーター型
/// @param lhs 入力vector
/// @param rhs 入力vector
/// @return 各要素を割った結果のvector
template <typename T, typename Alloc>
inline std::vector<T, Alloc> Divide(const std::vector<T, Alloc>& lhs,
                                    const std::vector<T, Alloc>& rhs) {
    ValidateSameSize(lhs, rhs, __func__);

    std::vector<T, Alloc> result(lhs.size());
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        result[i] = lhs[i] / rhs[i];
    }

    return result;
}

/// @brief 2つのvectorの各要素の剰余演算を行い，新しいvectorを返す関数
/// @tparam T vectorの要素の型
/// @tparam Alloc vectorのアロケーター型
/// @param lhs 入力vector
/// @param rhs 入力vector
/// @return 各要素の剰余演算結果のvector
template <typename T, typename Alloc>
inline std::vector<T, Alloc> Modulo(const std::vector<T, Alloc>& lhs,
                                    const std::vector<T, Alloc>& rhs) {
    ValidateSameSize(lhs, rhs, __func__);

    std::vector<T, Alloc> result(lhs.size());
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        result[i] = lhs[i] % rhs[i];
    }

    return result;
}

/// @brief operator+ のオーバーロード．Addの実装．
/// @tparam T vectorの要素の型
/// @tparam Alloc vectorのアロケーター型
/// @param lhs 入力vector
/// @param rhs 入力vector
/// @return 各要素を加算した結果のvector
template <typename T, typename Alloc>
inline std::vector<T, Alloc> operator+(const std::vector<T, Alloc>& lhs,
                                       const std::vector<T, Alloc>& rhs) {
    return Add(lhs, rhs);
}

/// @brief operator- のオーバーロード．Subtractの実装．
/// @tparam T vectorの要素の型
/// @tparam Alloc vectorのアロケーター型
/// @param lhs 入力vector
/// @param rhs 入力vector
/// @return 各要素を減算した結果のvector
template <typename T, typename Alloc>
inline std::vector<T, Alloc> operator-(const std::vector<T, Alloc>& lhs,
                                       const std::vector<T, Alloc>& rhs) {
    return Subtract(lhs, rhs);
}

/// @brief operator* のオーバーロード．Multiplyの実装．
/// @tparam T vectorの要素の型
/// @tparam Alloc vectorのアロケーター型
/// @param lhs 入力vector
/// @param rhs 入力vector
/// @return 各要素を乗算した結果のvector
template <typename T, typename Alloc>
inline std::vector<T, Alloc> operator*(const std::vector<T, Alloc>& lhs,
                                       const std::vector<T, Alloc>& rhs) {
    return Multiply(lhs, rhs);
}

/// @brief operator/ のオーバーロード．Divideの実装．
/// @tparam T vectorの要素の型
/// @tparam Alloc vectorのアロケーター型
/// @param lhs 入力vector
/// @param rhs 入力vector
/// @return 各要素を除算した結果のvector
template <typename T, typename Alloc>
inline std::vector<T, Alloc> operator/(const std::vector<T, Alloc>& lhs,
                                       const std::vector<T, Alloc>& rhs) {
    return Divide(lhs, rhs);
}

/// @brief operator% のオーバーロード．剰余演算を行う．
/// @tparam T vectorの要素の型
/// @tparam Alloc vectorのアロケーター型
/// @param lhs 入力vector
/// @param rhs 入力vector
/// @return 各要素の剰余演算結果のvector
template <typename T, typename Alloc>
inline std::vector<T, Alloc> operator%(const std::vector<T, Alloc>& lhs,
                                       const std::vector<T, Alloc>& rhs) {
    return Modulo(lhs, rhs);
}

/// @brief lhsにrhsを加算するin-place関数
/// @tparam T vectorの要素の型
/// @tparam Alloc vectorのアロケーター型
/// @param lhs 加算対象のvector（結果が直接書き換えられる）
/// @param rhs 加算するvector
template <typename T, typename Alloc>
inline void AddAssign(std::vector<T, Alloc>& lhs,
                      const std::vector<T, Alloc>& rhs) {
    ValidateSameSize(lhs, rhs, __func__);

    for (std::size_t i = 0; i < lhs.size(); ++i) {
        lhs[i] += rhs[i];
    }
}

/// @brief lhsからrhsを減算するin-place関数
/// @tparam T vectorの要素の型
/// @tparam Alloc vectorのアロケーター型
/// @param lhs 減算対象のvector（結果が直接書き換えられる）
/// @param rhs 減算するvector
template <typename T, typename Alloc>
inline void SubtractAssign(std::vector<T, Alloc>& lhs,
                           const std::vector<T, Alloc>& rhs) {
    ValidateSameSize(lhs, rhs, __func__);

    for (std::size_t i = 0; i < lhs.size(); ++i) {
        lhs[i] -= rhs[i];
    }
}

/// @brief lhsにrhsを乗算するin-place関数
/// @tparam T vectorの要素の型
/// @tparam Alloc vectorのアロケーター型
/// @param lhs 乗算対象のvector（結果が直接書き換えられる）
/// @param rhs 乗算するvector
template <typename T, typename Alloc>
inline void MultiplyAssign(std::vector<T, Alloc>& lhs,
                           const std::vector<T, Alloc>& rhs) {
    ValidateSameSize(lhs, rhs, __func__);

    for (std::size_t i = 0; i < lhs.size(); ++i) {
        lhs[i] *= rhs[i];
    }
}

/// @brief lhsをrhsで除算するin-place関数
/// @tparam T vectorの要素の型
/// @tparam Alloc vectorのアロケーター型
/// @param lhs 除算対象のvector（結果が直接書き換えられる）
/// @param rhs 除算するvector
template <typename T, typename Alloc>
inline void DivideAssign(std::vector<T, Alloc>& lhs,
                         const std::vector<T, Alloc>& rhs) {
    ValidateSameSize(lhs, rhs, __func__);

    for (std::size_t i = 0; i < lhs.size(); ++i) {
        lhs[i] /= rhs[i];
    }
}

/// @brief 浮動小数点型のvector比較用のデフォルト許容誤差（宣言）
/// @tparam T 浮動小数点型（float, double, long double）
template <typename T>
constexpr T defaultTolerance();

/// @brief float型のデフォルト許容誤差
template <>
constexpr float defaultTolerance<float>() {
    return 1e-5f;
}

/// @brief double型のデフォルト許容誤差
template <>
constexpr double defaultTolerance<double>() {
    return 1e-12;
}

/// @brief long double型のデフォルト許容誤差
template <>
constexpr long double defaultTolerance<long double>() {
    return 1e-14L;
}

/// @brief 2つのvectorが指定された許容誤差以内で"近い"かどうかを判定する関数
/// @tparam T vectorの要素の型（浮動小数点型に限定）
/// @tparam Alloc vectorのアロケーター型
/// @param lhs 比較対象のvector
/// @param rhs 比較対象のvector
/// @param tol 許容誤差（デフォルトはdefaultToleranceが使用される）
/// @return 全要素がtol以内であればtrue，そうでなければfalse
/// @pre Tは浮動小数点型であること
/// @details floatは1e-5f，doubleは1e-12，long
/// doubleは1e-14Lがデフォルト許容誤差
template <typename T, typename Alloc>
inline bool Close(const std::vector<T, Alloc>& lhs,
                  const std::vector<T, Alloc>& rhs,
                  T tol = defaultTolerance<T>()) {
    static_assert(std::is_floating_point<T>::value,
                  "T must be a floating point type");

    if (lhs.size() != rhs.size()) {
        return false;
    }

    for (std::size_t i = 0; i < lhs.size(); ++i) {
        if (std::abs(lhs[i] - rhs[i]) > tol) {
            return false;
        }
    }

    return true;
}

/// @brief ベクトルの2乗ノルム（Squared Norm）を計算する関数
/// @tparam T ベクトルの要素の型
/// @tparam Alloc ベクトルのアロケーター型
/// @param vec 入力ベクトル
/// @return 入力ベクトルの全要素の2乗の和
template <typename T, typename Alloc>
inline T SquaredNorm(const std::vector<T, Alloc>& vec) {
    T result = 0;
    for (const auto& elem : vec) {
        result += elem * elem;
    }

    return result;
}

/// @brief ベクトルのノルム（Norm）を計算する関数
/// @tparam T ベクトルの要素の型
/// @tparam Alloc ベクトルのアロケーター型
/// @param vec 入力ベクトル
/// @return 入力ベクトルのノルム
template <typename T, typename Alloc>
inline T Norm(const std::vector<T, Alloc>& vec) {
    return std::sqrt(SquaredNorm(vec));
}

/// @brief 2つのvector間の2乗ユークリッド距離（Squared Euclidean
/// Distance）を計算する関数
/// @tparam T vectorの要素の型
/// @tparam Alloc vectorのアロケーター型
/// @param lhs 入力vector
/// @param rhs 入力vector
/// @return 各要素の差の2乗の和（2乗ユークリッド距離）
template <typename T, typename Alloc>
inline T SquaredEuclideanDistance(const std::vector<T, Alloc>& lhs,
                                  const std::vector<T, Alloc>& rhs) {
    ValidateSameSize(lhs, rhs, __func__);

    T result = 0;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        const T diff = lhs[i] - rhs[i];
        result += diff * diff;
    }

    return result;
}

/// @brief 2つのvector間のユークリッド距離（Euclidean Distance）を計算する関数
/// @tparam T vectorの要素の型
/// @tparam Alloc vectorのアロケーター型
/// @param lhs 入力vector
/// @param rhs 入力vector
/// @return 各要素の差の2乗の和の平方根（ユークリッド距離）
template <typename T, typename Alloc>
inline T EuclideanDistance(const std::vector<T, Alloc>& lhs,
                           const std::vector<T, Alloc>& rhs) {
    return std::sqrt(SquaredEuclideanDistance(lhs, rhs));
}

/// @brief std::vectorの各要素を指定した区切り文字で連結した文字列を返す関数
/// @tparam T vectorの要素の型（operator<<が使える型に限定）
/// @tparam Alloc vectorのアロケーター型
/// @param vec 入力vector
/// @param delimiter 各要素の区切り文字列（デフォルトは","）
/// @return 区切り文字で連結された文字列
template <typename T, typename Alloc>
inline std::string Join(const std::vector<T, Alloc>& vec,
                        const std::string& delimiter = ",") {
    if (vec.empty()) {
        return "";
    }

    std::ostringstream oss;
    for (std::size_t i = 0; i < vec.size(); ++i) {
        if (i != 0) {
            oss << delimiter;
        }
        oss << vec[i];
    }

    return oss.str();
}

}  // namespace Eacpp