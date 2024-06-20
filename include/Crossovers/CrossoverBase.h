#ifndef CrossOverBase_H
#define CrossOverBase_H

#include <eigen3/Eigen/Core>
#include <vector>

#include "ICrossover.h"
#include "Utils/TemplateType.h"

namespace Eacpp {

template <Number T>
class CrossoverBase : public ICrossover<T> {
   public:
    CrossoverBase(int parentNum, int childrenNum) : _parentNum(parentNum), _childrenNum(childrenNum) {}
    int GetParentNum() override { return _parentNum; }
    int GetChildrenNum() override { return _childrenNum; }
    std::vector<Eigen::VectorX<T>> Cross(std::vector<Eigen::VectorX<T>> parents) override;

   protected:
    virtual std::vector<Eigen::VectorX<T>> performCrossover(std::vector<Eigen::VectorX<T>> parents) = 0;

   private:
    int _parentNum;
    int _childrenNum;
};

}  // namespace Eacpp

#endif