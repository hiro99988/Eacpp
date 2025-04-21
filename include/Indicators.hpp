#pragma once

#include <eigen3/Eigen/Core>
#include <vector>

namespace Eacpp {

class IGD {
   public:
    IGD(const std::vector<Eigen::ArrayXd>& paretoFront)
        : _paretoFront(paretoFront) {}
    IGD(const std::vector<std::vector<double>>& paretoFront);

    constexpr const char* Name() const {
        return _Name;
    }
    double Calculate(const std::vector<Eigen::ArrayXd>& objectives);
    double Calculate(const std::vector<std::vector<double>>& objectives);

   private:
    constexpr static const char* _Name = "IGD";
    std::vector<Eigen::ArrayXd> _paretoFront;
};

class IGDPlus {
   public:
    IGDPlus(const std::vector<Eigen::ArrayXd>& paretoFront)
        : _paretoFront(paretoFront) {}
    IGDPlus(const std::vector<std::vector<double>>& paretoFront);

    constexpr const char* Name() const {
        return _Name;
    }
    double Calculate(const std::vector<Eigen::ArrayXd>& objectives);
    double Calculate(const std::vector<std::vector<double>>& objectives);

   private:
    constexpr static const char* _Name = "IGD+";
    std::vector<Eigen::ArrayXd> _paretoFront;
};

}  // namespace Eacpp