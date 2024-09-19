#pragma once

#include <mpi.h>

#include <tuple>
#include <type_traits>
#include <vector>

namespace Eacpp {

/**
 * @brief 与えられたC++の型に対応するMPIデータ型を取得します。
 *
 * @tparam T C++の型。
 * @param var 型Tの変数。
 * @return MPI_Datatype 対応するMPIデータ型。
 */
template <typename T>
MPI_Datatype GetMpiDataType(std::vector<T> var) {
    if constexpr (std::is_same_v<signed char, T>) {
        return MPI_CHAR;
    } else if constexpr (std::is_same_v<signed short int, T>) {
        return MPI_SHORT;
    } else if constexpr (std::is_same_v<signed int, T>) {
        return MPI_INT;
    } else if constexpr (std::is_same_v<signed long int, T>) {
        return MPI_LONG;
    } else if constexpr (std::is_same_v<unsigned char, T>) {
        return MPI_UNSIGNED_CHAR;
    } else if constexpr (std::is_same_v<unsigned short int, T>) {
        return MPI_UNSIGNED_SHORT;
    } else if constexpr (std::is_same_v<unsigned int, T>) {
        return MPI_UNSIGNED;
    } else if constexpr (std::is_same_v<unsigned long int, T>) {
        return MPI_UNSIGNED_LONG;
    } else if constexpr (std::is_same_v<float, T>) {
        return MPI_FLOAT;
    } else if constexpr (std::is_same_v<double, T>) {
        return MPI_DOUBLE;
    } else if constexpr (std::is_same_v<long double, T>) {
        return MPI_LONG_DOUBLE;
    } else if constexpr (std::is_same_v<long long int, T>) {
        return MPI_LONG_LONG_INT;
    } else if constexpr (std::is_same_v<unsigned long long int, T>) {
        return MPI_UNSIGNED_LONG_LONG;
    } else if constexpr (std::is_same_v<std::pair<float, int>, T>) {
        return MPI_FLOAT_INT;
    } else if constexpr (std::is_same_v<std::pair<double, int>, T>) {
        return MPI_DOUBLE_INT;
    } else if constexpr (std::is_same_v<std::pair<long, int>, T>) {
        return MPI_LONG_INT;
    } else if constexpr (std::is_same_v<std::pair<int, int>, T>) {
        return MPI_2INT;
    } else if constexpr (std::is_same_v<std::pair<short, int>, T>) {
        return MPI_SHORT_INT;
    } else if constexpr (std::is_same_v<std::pair<long double, int>, T>) {
        return MPI_LONG_DOUBLE_INT;
    } else {
        static_assert(false, "Unsupported type");
    }
}

/**
 * @brief ノードのワークロードを計算します。
 *
 * @param totalTasks 全タスクの数。
 * @param rank ノードのランク。
 * @param parallelSize ノードの総数。
 * @return int ノードのワークロード。
 */
inline int CalculateNodeWorkload(int totalTasks, int rank, int parallelSize) {
    int nodeWorkload = totalTasks / parallelSize;
    if (rank < totalTasks % parallelSize) {
        nodeWorkload++;
    }
    return nodeWorkload;
}

/**
 * @brief すべてのノードのワークロードを計算します。
 *
 * @param totalTasks 全タスクの数。
 * @param parallelSize ノードの総数。
 * @return std::vector<int> すべてのノードのワークロード。
 */
inline std::vector<int> CalculateNodeWorkloads(int totalTasks, int parallelSize) {
    std::vector<int> nodeWorkloads(parallelSize);
    for (int i = 0; i < parallelSize; i++) {
        nodeWorkloads[i] = CalculateNodeWorkload(totalTasks, i, parallelSize);
    }
    return nodeWorkloads;
}

/**
 * @brief ノードの開始インデックスを計算します。
 *
 * @param totalTasks 全タスクの数。
 * @param rank ノードのランク。
 * @param parallelSize ノードの総数。
 * @return int ノードの開始インデックス。
 */
inline int CalculateNodeStartIndex(int totalTasks, int rank, int parallelSize) {
    int startIndex = 0;
    for (int i = 0; i < rank; i++) {
        int nodeWorkload = CalculateNodeWorkload(totalTasks, i, parallelSize);
        startIndex += nodeWorkload;
    }
    return startIndex;
}

/**
 * @brief
 * scatterv操作のための各ノードのデータ数と変位を生成します。
 *
 * @param nodeWorkloads すべてのノードのワークロード。
 * @param dataSize 各データ要素のサイズ。
 * @param parallelSize ノードの総数。
 * @return std::pair<std::vector<int>, std::vector<int>> データカウントとディスプレースメント。
 */
inline std::pair<std::vector<int>, std::vector<int>> GenerateDataCountsAndDisplacements(const std::vector<int>& nodeWorkloads,
                                                                                        int dataSize, int parallelSize) {
    std::vector<int> dataCounts(parallelSize);
    std::vector<int> displacements(parallelSize);
    for (int i = 0; i < parallelSize; i++) {
        dataCounts[i] = nodeWorkloads[i] * dataSize;
        displacements[i] = i == 0 ? 0 : displacements[i - 1] + dataCounts[i - 1];
    }
    return {dataCounts, displacements};
}

/**
 * @brief scatterv操作を使用して、送信ベクトルからすべてのノードにデータを分散します。
 *
 * @tparam T データの型。
 * @param send ルートノードから送信されるデータ。
 * @param nodeWorkloads すべてのノードのワークロード。
 * @param dataSize 各データ要素のサイズ。
 * @param rank 現在のノードのランク。
 * @param parallelSize ノードの総数。
 * @return std::vector<T> 現在のノードで受信されたデータ。
 */
template <typename T>
std::vector<T> Scatterv(const std::vector<T>& send, const std::vector<int>& nodeWorkloads, int dataSize, int rank,
                        int parallelSize) {
    std::vector<int> dataCounts;
    std::vector<int> displacements;
    if (rank == 0) {
        std::tie(dataCounts, displacements) = GenerateDataCountsAndDisplacements(nodeWorkloads, dataSize, parallelSize);
    }
    int receivedDataCount;
    MPI_Scatter(dataCounts.data(), 1, MPI_INT, &receivedDataCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<T> received(receivedDataCount);
    MPI_Scatterv(send.data(), dataCounts.data(), displacements.data(), GetMpiDataType(send), received.data(), receivedDataCount,
                 GetMpiDataType(received), 0, MPI_COMM_WORLD);
    return received;
}

int GetRankFromIndex(int totalTasks, int index, int parallelSize) {
    int start = 0;
    int end = 0;
    int tasksPerNode = totalTasks / parallelSize;
    int tasksRemainder = totalTasks % parallelSize;
    int loop = parallelSize - 1;
    for (int i = 0; i < loop; i++) {
        start = end;
        end += tasksPerNode;
        if (i < tasksRemainder) {
            end++;
        }
        if (start <= index && index < end) {
            return i;
        }
    }
    return parallelSize - 1;
}

}  // namespace Eacpp
