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
        int size = parents[0].size();
        // // int crossoverPoint = _rng.Integer(1, size - 1);
        int crossoverPoint = 1;
        Eigen::VectorX<T> child(size);
        child << parents[0].head(crossoverPoint), parents[1].tail(size - crossoverPoint);
        return {child};
    }
};

int main() {
    srand(time(0));
    // Rng rng;
    // cout << rng.Integers(10) << endl;
    // cout << rng.Integers(5, 10) << endl;
    // auto v = rng.Integers(5, 10, 10, true);
    // for (auto i : v) {
    //     cout << i << " ";
    // }
    // cout << endl;

    OnePointCrossover<int> op;
    Eigen::VectorXi v1 = Eigen::VectorXi::Random(4);
    Eigen::VectorXi v2 = Eigen::VectorXi::Random(4);
    vector<Eigen::VectorXi> parents = {v1, v2};
    auto child = op.Cross(parents);
    cout << v1 << endl
         << endl;
    cout << v2 << endl
         << endl;
    cout << child[0] << endl;

    return 0;
}