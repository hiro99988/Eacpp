#pragma once

#include <gmock/gmock.h>

#include <Eigen/Core>

#include "Individual.h"
#include "Mutations/IMutation.h"

namespace Eacpp {

template <typename T>
class MockMutation : public IMutation<T> {
   public:
    MOCK_CONST_METHOD1_T(Mutate, void(Individual<T>&));
};

}  // namespace Eacpp