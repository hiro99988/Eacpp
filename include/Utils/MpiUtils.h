#pragma once

#include <gtest/gtest.h>
#include <mpi.h>

#include <tuple>
#include <type_traits>
#include <vector>

namespace Eacpp {

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

inline int CalculateNodeWorkload(int totalTasks, int rank, int parallelSize) {
    int nodeWorkload = totalTasks / parallelSize;
    if (rank < totalTasks % parallelSize) {
        nodeWorkload++;
    }
    return nodeWorkload;
}

inline std::vector<int> CalculateNodeWorkloads(int totalTasks, int parallelSize) {
    std::vector<int> nodeWorkloads(parallelSize);
    for (int i = 0; i < parallelSize; i++) {
        nodeWorkloads[i] = CalculateNodeWorkload(totalTasks, i, parallelSize);
    }
    return nodeWorkloads;
}

inline int CalculateNodeStartIndex(int totalTasks, int rank, int parallelSize) {
    int startIndex = 0;
    for (int i = 0; i < rank; i++) {
        int nodeWorkload = CalculateNodeWorkload(totalTasks, i, parallelSize);
        startIndex += nodeWorkload;
    }
    return startIndex;
}

inline std::pair<std::vector<int>, std::vector<int>> GenerateDataCountsAndDisplacements(std::vector<int>& nodeWorkloads,
                                                                                        int dataSize, int parallelSize) {
    std::vector<int> dataCounts(parallelSize);
    std::vector<int> displacements(parallelSize);
    for (int i = 0; i < parallelSize; i++) {
        dataCounts[i] = nodeWorkloads[i] * dataSize;
        displacements[i] = i == 0 ? 0 : displacements[i - 1] + dataCounts[i - 1];
    }
    return {dataCounts, displacements};
}

template <typename T>
std::vector<T> Scatterv(std::vector<T>& send, std::vector<int>& nodeWorkloads, int dataCount, int dataSize, int rank,
                        int parallelSize) {
    std::vector<int> dataCounts;
    std::vector<int> displacements;
    if (rank == 0) {
        std::tie(dataCounts, displacements) = GenerateDataCountsAndDisplacements(nodeWorkloads, dataSize, parallelSize);
    }
    int receivedDataCount = dataCount * dataSize;
    std::vector<T> received(receivedDataCount);
    MPI_Scatterv(send.data(), dataCounts.data(), displacements.data(), GetMpiDataType(send), received.data(), receivedDataCount,
                 GetMpiDataType(received), 0, MPI_COMM_WORLD);
    return received;
}

template <typename T>
std::vector<T> Scatterv(std::vector<T>& send, int dataCount, int dataSize, int rank, int parallelSize) {
    std::vector<int> nodeWorkloads;
    if (rank == 0) {
        nodeWorkloads = CalculateNodeWorkloads(send.size(), parallelSize);
    }
    return Scatterv(send, nodeWorkloads, dataCount, dataSize, rank, parallelSize);
}

}  // namespace Eacpp
