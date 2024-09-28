#include <gtest/gtest.h>
#include <mpi.h>

#include "Utils/MpiUtils.h"

// -n 4 で実行すること。

namespace Eacpp::Test {

class MpiUtilsTestM : public ::testing::Test {
   protected:
    static void SetUpTestSuite() { MPI_Init(nullptr, nullptr); }

    static void TearDownTestSuite() { MPI_Finalize(); }
};

TEST_F(MpiUtilsTestM, Scatterv) {
    int rank, parallelSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &parallelSize);
    std::vector<int> send = {0, 1, 2, 3, 4};
    std::vector<int> nodeWorkloads = CalculateNodeWorkloads(send.size(), parallelSize);
    std::vector<int> received = Scatterv(send, nodeWorkloads, 1, rank, parallelSize);

    if (rank == 0) {
        ASSERT_EQ(received.size(), 2);
        ASSERT_EQ(received[0], 0);
        ASSERT_EQ(received[1], 1);
    } else if (rank == 1) {
        ASSERT_EQ(received.size(), 1);
        ASSERT_EQ(received[0], 2);
    } else if (rank == 2) {
        ASSERT_EQ(received.size(), 1);
        ASSERT_EQ(received[0], 3);
    } else if (rank == 3) {
        ASSERT_EQ(received.size(), 1);
        ASSERT_EQ(received[0], 4);
    }
}

}  // namespace Eacpp::Test