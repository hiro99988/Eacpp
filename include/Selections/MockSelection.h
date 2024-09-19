#pragma once

#include <gmock/gmock.h>

#include <vector>

#include "Selections/ISelection.h"

namespace Eacpp {

class MockSelection : public ISelection {
   public:
    MOCK_CONST_METHOD2(Select, std::vector<int>(int, const std::vector<int>&));
};

}  // namespace Eacpp