#ifndef CrossOverBase_H
#define CrossOverBase_H

#include <eigen3/Eigen/Core>

#include "ICrossover.h"
#include "Utils/TemplateType.h"

namespace Eacpp {

template <Number T>
class CrossoverBase : public ICrossover<T> {
   public:
    CrossoverBase(int parentNum, int childrenNum) : _parentNum(parentNum), _childrenNum(childrenNum) {}
    virtual ~CrossoverBase() {}

    int GetParentNum() const override { return _parentNum; }
    int GetChildrenNum() const override { return _childrenNum; }

    Eigen::ArrayXX<T> Cross(const Eigen::ArrayXX<T>& parents) const override;

   protected:
    virtual Eigen::ArrayXX<T> performCrossover(const Eigen::ArrayXX<T>& parents) const = 0;

   private:
    int _parentNum;
    int _childrenNum;
};

}  // namespace Eacpp

#endif