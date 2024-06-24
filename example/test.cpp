#include <concepts>
#include <cstdint>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <random>
#include <vector>

// #include "Crossovers/OnePointCrossover.h"
// #include "Utils/Rng.h"

using namespace std;
// using namespace Eacpp;

template <typename T>
class OnePointCrossover {
   public:
    OnePointCrossover() {}
    std::vector<Eigen::VectorX<T>> Cross(std::vector<Eigen::VectorX<T>> parents) {
        throw std::runtime_error("Not implemented");
    }
};

void swapIfMaxLessThanMin(int& min, int& max) {
    if (max < min) {
        std::swap(min, max);
    }
}

Eigen::MatrixXd Mutate() { return Eigen::MatrixXd::Random(10, 10); }

int main() {
    int min = 100;
    int max = 10;
    swapIfMaxLessThanMin(min, max);
    cout << min << " " << max << endl;

    return 0;
}