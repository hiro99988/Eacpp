#include <gtest/gtest.h>
#include <mpi.h>

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

}  // namespace Eacpp
