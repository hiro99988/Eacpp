#include <iostream>

#include "Individual.h"

using namespace Eacpp;

int main() {
    Individuali individual(Eigen::ArrayXi::LinSpaced(3, 0, 2),
                           Eigen::ArrayXd::Constant(2, 1.0),
                           Eigen::ArrayXd::Constant(2, 0.5), {1, 2});
    Individuali other;
    other = individual;
    std::cout << individual << std::endl;
    std::cout << other << std::endl;

    return 0;
}