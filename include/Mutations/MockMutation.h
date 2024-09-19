#pragma once

#include <gmock/gmock.h>

#include <eigen3/Eigen/Core>

#include "Mutations/IMutation.h"

namespace Eacpp {

template <typename T>
class MockMutation : public IMutation<T> {
   public:
    MOCK_CONST_METHOD1_T(Mutate, void(Eigen::ArrayX<T>&));
};

}  // namespace Eacpp