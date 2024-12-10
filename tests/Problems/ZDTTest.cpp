#include <gtest/gtest.h>

#include <eigen3/Eigen/Core>

#include "Problems/ZDT1.h"
#include "Problems/ZDT2.h"
#include "Problems/ZDT3.h"
#include "Problems/ZDT4.h"
#include "Problems/ZDT6.h"
#include "Problems/ZDTBase.h"

namespace Eacpp {

class ZDT1Test : public ::testing::Test {
   protected:
    void SetUp() override {}
    void TearDown() override {}

    ZDT1 zdt{10};
    static constexpr std::array<std::array<double, 2>, 10> F = {{{0.417022004702574, 2.4821992632522214},
                                                                 {0.4191945144032948, 3.3645698036049874},
                                                                 {0.8007445686755367, 3.740366073594546},
                                                                 {0.0983468338330501, 5.427675486264629},
                                                                 {0.9888610889064947, 2.7678850194472897},
                                                                 {0.019366957870297075, 4.408804850291295},
                                                                 {0.10233442882782584, 5.04668787619817},
                                                                 {0.9034019152878835, 3.190706357014327},
                                                                 {0.0, 1.0},
                                                                 {1.0, 6.83772233983162}}};
};

class ZDT2Test : public ::testing::Test {
   protected:
    void SetUp() override {}
    void TearDown() override {}

    ZDT2 zdt{10};
    static constexpr std::array<std::array<double, 2>, 10> F = {{{0.417022004702574, 3.6826380049007597},
                                                                 {0.4191945144032948, 4.743365976128229},
                                                                 {0.8007445686755367, 5.8087131457832095},
                                                                 {0.0983468338330501, 6.207556650820672},
                                                                 {0.9888610889064947, 4.793022485239073},
                                                                 {0.019366957870297075, 4.710776469884046},
                                                                 {0.10233442882782584, 5.816518943459582},
                                                                 {0.9034019152878835, 5.248101015125405},
                                                                 {0.0, 1.0},
                                                                 {1.0, 9.9}}};
};

class ZDT3Test : public ::testing::Test {
   protected:
    void SetUp() override {}
    void TearDown() override {}

    ZDT3 zdt{10};
    static constexpr std::array<std::array<double, 2>, 10> F = {{{0.417022004702574, 2.269669701992888},
                                                                 {0.4191945144032948, 3.1268332690547864},
                                                                 {0.8007445686755367, 3.7216373133321436},
                                                                 {0.0983468338330501, 5.422570065881676},
                                                                 {0.9888610889064947, 3.106906852117654},
                                                                 {0.019366957870297075, 4.397735071598455},
                                                                 {0.10233442882782584, 5.054186177746011},
                                                                 {0.9034019152878835, 3.28707312618877},
                                                                 {0.0, 1.0},
                                                                 {1.0, 6.837722339831621}}};
};

class ZDT4Test : public ::testing::Test {
   protected:
    void SetUp() override {}
    void TearDown() override {}

    ZDT4 zdt{10};
    static constexpr std::array<std::array<double, 2>, 10> F = {{{0.417022004702574, 90.85247661260021},
                                                                 {0.4191945144032948, 95.0262100495323},
                                                                 {0.8007445686755367, 80.41657132251582},
                                                                 {0.0983468338330501, 94.68621905507656},
                                                                 {0.9888610889064947, 113.7813676944705},
                                                                 {0.019366957870297075, 99.2296126674139},
                                                                 {0.10233442882782584, 55.29069455243222},
                                                                 {0.9034019152878835, 113.05480475604514},
                                                                 {0.0, 1.0},
                                                                 {1.0, 6.83772233983162}}};
};

class ZDT6Test : public ::testing::Test {
   protected:
    void SetUp() override {}
    void TearDown() override {}

    ZDT6 zdt{10};
    static constexpr std::array<std::array<double, 2>, 10> F = {{{0.8114180480067016, 7.592981519342151},
                                                                 {0.8142942529478623, 8.164917108114777},
                                                                 {0.9985099955067418, 8.623535128893558},
                                                                 {0.4710851664519831, 8.824982135428707},
                                                                 {0.9999984301377043, 8.223567573383889},
                                                                 {0.9980840040725161, 8.090603725701571},
                                                                 {0.5518821505885969, 8.66347926643958},
                                                                 {0.9776995203805138, 8.413226600413248},
                                                                 {1.0, 0.0},
                                                                 {1.0, 9.9}}};
};

}  // namespace Eacpp

namespace Eacpp::Test {

constexpr std::array<std::array<double, 10>, 10> X = {
    {{0.417022004702574, 0.7203244934421581, 0.00011437481734488664, 0.30233257263183977, 0.14675589081711304,
      0.0923385947687978, 0.1862602113776709, 0.34556072704304774, 0.39676747423066994, 0.538816734003357},
     {0.4191945144032948, 0.6852195003967595, 0.20445224973151743, 0.8781174363909454, 0.027387593197926163, 0.6704675101784022,
      0.41730480236712697, 0.5586898284457517, 0.14038693859523377, 0.1981014890848788},
     {0.8007445686755367, 0.9682615757193975, 0.31342417815924284, 0.6923226156693141, 0.8763891522960383, 0.8946066635038473,
      0.08504421136977791, 0.03905478323288236, 0.1698304195645689, 0.8781425034294131},
     {0.0983468338330501, 0.42110762500505217, 0.9578895301505019, 0.5331652849730171, 0.6918771139504734, 0.31551563100606295,
      0.6865009276815837, 0.8346256718973729, 0.018288277344191806, 0.7501443149449675},
     {0.9888610889064947, 0.7481656543798394, 0.2804439920644052, 0.7892793284514885, 0.10322600657764203, 0.44789352617590517,
      0.9085955030930956, 0.2936141483736795, 0.28777533858634874, 0.13002857211827767},
     {0.019366957870297075, 0.678835532939891, 0.21162811600005904, 0.2655466593722262, 0.4915731592803383,
      0.053362545117080384, 0.5741176054920131, 0.14672857490581015, 0.5893055369032842, 0.6997583600209312},
     {0.10233442882782584, 0.4140559878195683, 0.6944001577277451, 0.41417926952690265, 0.04995345894608716, 0.5358964059155116,
      0.6637946452197888, 0.5148891120583086, 0.9445947559908133, 0.5865550405019929},
     {0.9034019152878835, 0.13747470414623753, 0.13927634725075855, 0.8073912887095238, 0.3976768369855336, 0.16535419711693278,
      0.9275085803960339, 0.34776585974550656, 0.7508121031361555, 0.7259979853504515},
     {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
     {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}}};

class ZDTBaseTest : public ::testing::Test {
   protected:
    void SetUp() override {}
    void TearDown() override {}

    class ZDTBaseTmp : public ZDTBase {
       public:
        ZDTBaseTmp(int decisionVariablesNum) : ZDTBase(decisionVariablesNum) {}
        double F1(double x1) const override {
            return 1.0;
        }
        double G(const Eigen::ArrayXd& X) const override {
            return 1.0;
        }
        double H(double f1, double g) const override {
            return 2.0;
        }
    };

    ZDTBaseTmp zdtBaseTmp{1};
};

TEST_F(ZDTBaseTest, ComputeObjectiveSet) {
    Individuald individual(1);
    zdtBaseTmp.ComputeObjectiveSet(individual);
    Eigen::ArrayXd expected(2);
    expected << 1.0, 2.0;
    ASSERT_TRUE((individual.objectives == expected).all());
}

TEST_F(ZDTBaseTest, IsFeasible) {
    Individuald individual(1);
    individual.solution << 0.0;
    ASSERT_TRUE(zdtBaseTmp.IsFeasible(individual));
    individual.solution << -0.1;
    ASSERT_FALSE(zdtBaseTmp.IsFeasible(individual));
    individual.solution << 1.0;
    ASSERT_TRUE(zdtBaseTmp.IsFeasible(individual));
    individual.solution << 1.1;
    ASSERT_FALSE(zdtBaseTmp.IsFeasible(individual));
}

TEST_F(ZDTBaseTest, EvaluateConstraints) {
    Individuald individual(1);
    individual.solution << 0.0;
    std::vector<bool> expected = {true};
    ASSERT_EQ(zdtBaseTmp.EvaluateConstraints(individual), expected);
    individual.solution << -0.1;
    expected = {false};
    ASSERT_EQ(zdtBaseTmp.EvaluateConstraints(individual), expected);
    individual.solution << 1.0;
    expected = {true};
    ASSERT_EQ(zdtBaseTmp.EvaluateConstraints(individual), expected);
    individual.solution << 1.1;
    expected = {false};
    ASSERT_EQ(zdtBaseTmp.EvaluateConstraints(individual), expected);
}

TEST_F(ZDT1Test, ComputeObjectiveSet) {
    for (int i = 0; i < X.size(); ++i) {
        Individuald individual{Eigen::Map<const Eigen::ArrayXd>(X[i].data(), X[i].size())};
        zdt.ComputeObjectiveSet(individual);
        for (int j = 0; j < F[i].size(); ++j) {
            EXPECT_DOUBLE_EQ(individual.objectives(j), F[i][j]);
        }
    }
}

TEST_F(ZDT2Test, ComputeObjectiveSet) {
    for (int i = 0; i < X.size(); ++i) {
        Individuald individual{Eigen::Map<const Eigen::ArrayXd>(X[i].data(), X[i].size())};
        zdt.ComputeObjectiveSet(individual);
        for (int j = 0; j < F[i].size(); ++j) {
            EXPECT_DOUBLE_EQ(individual.objectives(j), F[i][j]);
        }
    }
}

TEST_F(ZDT3Test, ComputeObjectiveSet) {
    for (int i = 0; i < X.size(); ++i) {
        Individuald individual{Eigen::Map<const Eigen::ArrayXd>(X[i].data(), X[i].size())};
        zdt.ComputeObjectiveSet(individual);
        for (int j = 0; j < F[i].size(); ++j) {
            EXPECT_DOUBLE_EQ(individual.objectives(j), F[i][j]);
        }
    }
}

TEST_F(ZDT4Test, ComputeObjectiveSet) {
    for (int i = 0; i < X.size(); ++i) {
        Individuald individual{Eigen::Map<const Eigen::ArrayXd>(X[i].data(), X[i].size())};
        zdt.ComputeObjectiveSet(individual);
        for (int j = 0; j < F[i].size(); ++j) {
            EXPECT_DOUBLE_EQ(individual.objectives(j), F[i][j]);
        }
    }
}

TEST_F(ZDT6Test, ComputeObjectiveSet) {
    for (int i = 0; i < X.size(); ++i) {
        Individuald individual{Eigen::Map<const Eigen::ArrayXd>(X[i].data(), X[i].size())};
        zdt.ComputeObjectiveSet(individual);
        for (int j = 0; j < F[i].size(); ++j) {
            EXPECT_DOUBLE_EQ(individual.objectives(j), F[i][j]);
        }
    }
}

}  // namespace Eacpp::Test