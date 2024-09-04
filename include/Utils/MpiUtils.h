#include <gtest/gtest.h>
#include <mpi.h>

#include <type_traits>

namespace Eacpp {

class MpiEnvironment : public ::testing::Environment {
   public:
    virtual ~MpiEnvironment() {}

    virtual void SetUp() {
        int argc = 0;
        char** argv;
        int mpiError = MPI_Init(&argc, &argv);
        ASSERT_FALSE(mpiError);
    }

    virtual void TearDown() {
        int mpiError = MPI_Finalize();
        ASSERT_FALSE(mpiError);
    }
};

template <typename T>
MPI_Datatype GetMpiDataType(T var) {
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

}  // namespace Eacpp
