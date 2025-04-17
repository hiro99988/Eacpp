#pragma once

#include <Eigen/Core>
#include <vector>

namespace Eacpp {

class IGD {
   public:
    IGD(const std::vector<Eigen::ArrayXd>& paretoFront)
        : _paretoFront(paretoFront) {}
    IGD(const std::vector<std::vector<double>>& paretoFront);

    double Calculate(const std::vector<Eigen::ArrayXd>& objectives);
    double Calculate(const std::vector<std::vector<double>>& objectives);

   private:
    std::vector<Eigen::ArrayXd> _paretoFront;
};

class IGDPlus {
   public:
    IGDPlus(const std::vector<Eigen::ArrayXd>& paretoFront)
        : _paretoFront(paretoFront) {}
    IGDPlus(const std::vector<std::vector<double>>& paretoFront);

    double Calculate(const std::vector<Eigen::ArrayXd>& objectives);
    double Calculate(const std::vector<std::vector<double>>& objectives);

   private:
    std::vector<Eigen::ArrayXd> _paretoFront;
};

}  // namespace Eacpp