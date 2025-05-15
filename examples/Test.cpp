#include <Eigen/Core>
#include <toml.hpp>

#include "Individual.h"
#include "Problems/Hpa.hpp"

using namespace Eacpp;
namespace py = pybind11;

int main(int argc, char* argv[]) {
    constexpr const char* config_file = "data/inputs/benchmarkParameter.toml";

    // Load the TOML configuration file
    auto config = toml::parse(config_file);
    int trial = toml::find<int>(config, "trial");
    int generationNum = toml::find<int>(config, "generation_num");
    std::cout << "trial: " << trial << std::endl;
    std::cout << "generationNum: " << generationNum << std::endl;

    return 0;
}