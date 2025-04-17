#define _TEST_

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <array>
#include <iostream>
#include <tuple>
#include <vector>

#include "Individual.h"
#include "Problems/DTLZ.hpp"

namespace Eacpp::Test {

constexpr std::array<std::array<double, 10>, 10> X = {
    {{0.417022004702574, 0.7203244934421581, 0.00011437481734488664,
      0.30233257263183977, 0.14675589081711304, 0.0923385947687978,
      0.1862602113776709, 0.34556072704304774, 0.39676747423066994,
      0.538816734003357},
     {0.4191945144032948, 0.6852195003967595, 0.20445224973151743,
      0.8781174363909454, 0.027387593197926163, 0.6704675101784022,
      0.41730480236712697, 0.5586898284457517, 0.14038693859523377,
      0.1981014890848788},
     {0.8007445686755367, 0.9682615757193975, 0.31342417815924284,
      0.6923226156693141, 0.8763891522960383, 0.8946066635038473,
      0.08504421136977791, 0.03905478323288236, 0.1698304195645689,
      0.8781425034294131},
     {0.0983468338330501, 0.42110762500505217, 0.9578895301505019,
      0.5331652849730171, 0.6918771139504734, 0.31551563100606295,
      0.6865009276815837, 0.8346256718973729, 0.018288277344191806,
      0.7501443149449675},
     {0.9888610889064947, 0.7481656543798394, 0.2804439920644052,
      0.7892793284514885, 0.10322600657764203, 0.44789352617590517,
      0.9085955030930956, 0.2936141483736795, 0.28777533858634874,
      0.13002857211827767},
     {0.019366957870297075, 0.678835532939891, 0.21162811600005904,
      0.2655466593722262, 0.4915731592803383, 0.053362545117080384,
      0.5741176054920131, 0.14672857490581015, 0.5893055369032842,
      0.6997583600209312},
     {0.10233442882782584, 0.4140559878195683, 0.6944001577277451,
      0.41417926952690265, 0.04995345894608716, 0.5358964059155116,
      0.6637946452197888, 0.5148891120583086, 0.9445947559908133,
      0.5865550405019929},
     {0.9034019152878835, 0.13747470414623753, 0.13927634725075855,
      0.8073912887095238, 0.3976768369855336, 0.16535419711693278,
      0.9275085803960339, 0.34776585974550656, 0.7508121031361555,
      0.7259979853504515},
     {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
     {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}}};

class DTLZ1Test : public ::testing::Test {
   protected:
    void SetUp() override {}
    void TearDown() override {}

    DTLZ1 dtlz{10, 3};
    std::vector<std::vector<double>> F = {
        {103.98274008198233, 40.37267338603425, 201.80237158932806},
        {118.02765095452536, 54.22029424578542, 238.65424765280477},
        {261.30214527259676, 8.565163133662042, 67.15315849822493},
        {18.983405287458456, 26.096294438308103, 413.2949928677151},
        {199.80845022641276, 67.2560014183712, 3.0083165537402228},
        {5.066392778172482, 2.396965476265117, 377.9021857004074},
        {18.8323723731736, 26.650298877024852, 398.9686416771594},
        {64.29031160244811, 403.3616248153226, 50.00463315984185},
        {0.0, 0.0, 100.5},
        {100.5, 0.0, 0.0}};

    int DecisionVariablesNum() const {
        return dtlz.DecisionVariablesNum();
    }
    int ObjectivesNum() const {
        return dtlz.ObjectivesNum();
    }
    const std::vector<std::pair<double, double>>& VariableBounds() const {
        return dtlz.VariableBounds();
    }
    void ComputeObjectiveSet(Individuald& individual) const {
        dtlz.ComputeObjectiveSet(individual);
    }
    bool IsFeasible(const Individuald& individual) const {
        return dtlz.IsFeasible(individual);
    }
    std::vector<bool> EvaluateConstraints(const Individuald& individual) const {
        return dtlz.EvaluateConstraints(individual);
    }
};

class DTLZ2Test : public ::testing::Test {
   protected:
    void SetUp() override {}
    void TearDown() override {}

    DTLZ2 dtlz{10, 3};
    std::vector<std::vector<double>> F = {
        {0.578229429594233, 1.2304276116901565, 1.044404440123023},
        {0.6431381144251349, 1.192928093032725, 1.048496323898157},
        {0.03077786601636937, 0.616841020683118, 1.9083948764331662},
        {1.3433982559969626, 1.0457951943166015, 0.2651148641681172},
        {0.011350132356563443, 0.027179868142067878, 1.6832379370631856},
        {0.7323585777839609, 1.3264025153929166, 0.04610756294309486},
        {1.1636008082433111, 0.8853047362380847, 0.23707271249737585},
        {0.24612045269098146, 0.05399027006861668, 1.647836006511234},
        {3.0, 0.0, 0.0},
        {1.1248198369963932e-32, 1.8369701987210297e-16, 3.0}};

    void ComputeObjectiveSet(Individuald& individual) const {
        dtlz.ComputeObjectiveSet(individual);
    }
};

class DTLZ3Test : public ::testing::Test {
   protected:
    void SetUp() override {}
    void TearDown() override {}

    DTLZ3 dtlz{10, 3};
    std::vector<std::vector<double>> F = {
        {233.5062116999207, 496.88320184319974, 421.76152200838465},
        {308.4544796289989, 572.1384037394917, 502.86770559385275},
        {10.342548592772118, 207.2823446251023, 641.2941928205738},
        {714.7806886420267, 556.4352981963619, 141.05942473380185},
        {3.641662760761106, 8.72059554425624, 540.0628574484135},
        {372.3649343635002, 674.40431581804, 23.44321507220896},
        {698.306920486806, 531.2942545890113, 142.27346236156941},
        {152.85744181952873, 33.53160810327335, 1023.417573547496},
        {201.0, 0.0, 0.0},
        {7.536292907875834e-31, 1.23077003314309e-14, 201.0}};

    void ComputeObjectiveSet(Individuald& individual) const {
        dtlz.ComputeObjectiveSet(individual);
    }
};

class DTLZ4Test : public ::testing::Test {
   protected:
    void SetUp() override {}
    void TearDown() override {}

    DTLZ4 dtlz{10, 3};
    std::vector<std::vector<double>> F = {
        {1.7143751093057493, 1.5242070952292778e-14, 2.7933341971441054e-38},
        {1.713490183981502, 1.0303217946909519e-16, 4.6941890885549156e-38},
        {2.0019366845509836, 0.12514246792870995, 7.043965866142474e-10},
        {1.7229893655720159, 7.442399882775089e-38, 5.110233671720184e-101},
        {1.4672463825324993, 5.786282927292955e-13, 0.825436597474788},
        {1.5158557407137903, 3.5746575516846796e-17, 1.2103710612458404e-171},
        {1.4811937037271443, 1.1820577269899334e-38, 2.338345749509208e-99},
        {1.6669894182514653, 1.7391301843532837e-86, 0.00010142696196578246},
        {3.0, 0.0, 0.0},
        {1.1248198369963932e-32, 1.8369701987210297e-16, 3.0}};

    void ComputeObjectiveSet(Individuald& individual) const {
        dtlz.ComputeObjectiveSet(individual);
    }
};

class DTLZ5Test : public ::testing::Test {
   protected:
    void SetUp() override {}
    void TearDown() override {}

    DTLZ5 dtlz{10, 3};
    std::vector<std::vector<double>> F = {
        {0.8131929080076364, 1.0895038665486683, 1.044404440123023},
        {0.8354707967137581, 1.0670954114846012, 1.048496323898157},
        {0.24989195140929032, 0.564795657255586, 1.9083948764331662},
        {1.2647723605202437, 1.139630351796331, 0.2651148641681172},
        {0.01728469012130806, 0.023849742643099112, 1.6832379370631856},
        {0.9642197586298229, 1.168748465822415, 0.04610756294309486},
        {1.078193381788643, 0.9875375174732558, 0.23707271249737585},
        {0.2138123237940208, 0.1333211036884131, 1.647836006511234},
        {2.897777478867205, 0.7764571353075622, 0.0},
        {4.754428727147651e-17, 1.7743769570680044e-16, 3.0}};

    void ComputeObjectiveSet(Individuald& individual) const {
        dtlz.ComputeObjectiveSet(individual);
    }
};

class DTLZ6Test : public ::testing::Test {
   protected:
    void SetUp() override {}
    void TearDown() override {}

    DTLZ6 dtlz{10, 3};
    std::vector<std::vector<double>> F = {
        {2.775362246805014, 5.260703899669196, 4.569269264285234},
        {3.2148637293703004, 5.478304232294132, 4.914211409619661},
        {0.3505798552303619, 2.4877365170005334, 7.762999870852047},
        {6.455292337238957, 5.180422354303647, 1.288913375256894},
        {0.061178177651943895, 0.12880934988544443, 8.149126752867181},
        {4.153979073579799, 6.942640411256371, 0.24620073798369368},
        {6.5160985710836705, 5.125094349259213, 1.3442064101294626},
        {1.215914166036182, 0.35478351694599725, 8.283347094251562},
        {0.7071067811865476, 0.7071067811865476, 0.0},
        {4.803075062245348e-17, 5.48993991755529e-16, 9.0}};

    void ComputeObjectiveSet(Individuald& individual) const {
        dtlz.ComputeObjectiveSet(individual);
    }
};

class DTLZ7Test : public ::testing::Test {
   protected:
    void SetUp() override {}
    void TearDown() override {}

    DTLZ7 dtlz{10, 3};
    std::vector<std::vector<double>> F = {
        {0.417022004702574, 0.7203244934421581, 11.58976537719187},
        {0.4191945144032948, 0.6852195003967595, 15.525087032615678},
        {0.8007445686755367, 0.9682615757193975, 16.50963813013696},
        {0.0983468338330501, 0.42110762500505217, 21.871381092824205},
        {0.9888610889064947, 0.7481656543798394, 14.57743353783027},
        {0.019366957870297075, 0.678835532939891, 15.45366713968852},
        {0.10233442882782584, 0.4140559878195683, 20.549393122295992},
        {0.9034019152878835, 0.13747470414623753, 18.496867820839846},
        {0.0, 0.0, 6.0},
        {1.0, 1.0, 30.999999999999996}};

    void ComputeObjectiveSet(Individuald& individual) const {
        dtlz.ComputeObjectiveSet(individual);
    }
};

TEST_F(DTLZ1Test, DecisionVariablesNum) {
    EXPECT_EQ(DecisionVariablesNum(), 10);
}

TEST_F(DTLZ1Test, ObjectivesNum) {
    EXPECT_EQ(ObjectivesNum(), 3);
}

TEST_F(DTLZ1Test, VariableBounds) {
    auto bounds = VariableBounds();
    EXPECT_EQ(bounds.size(), 1);
    EXPECT_DOUBLE_EQ(bounds[0].first, 0.0);
    EXPECT_DOUBLE_EQ(bounds[0].second, 1.0);
}

TEST_F(DTLZ1Test, ComputeObjectiveSet) {
    for (int i = 0; i < X.size(); ++i) {
        Individuald individual{
            Eigen::Map<const Eigen::ArrayXd>(X[i].data(), X[i].size())};
        ComputeObjectiveSet(individual);
        for (int j = 0; j < F[i].size(); ++j) {
            EXPECT_DOUBLE_EQ(individual.objectives(j), F[i][j]);
        }
    }
}

TEST_F(DTLZ1Test, IsFeasible) {
    Individuald individual{Eigen::ArrayXd::Zero(10)};
    EXPECT_TRUE(IsFeasible(individual));

    individual.solution = Eigen::ArrayXd::Ones(10);
    EXPECT_TRUE(IsFeasible(individual));

    individual.solution = Eigen::ArrayXd::Constant(10, 0.5);
    EXPECT_TRUE(IsFeasible(individual));

    individual.solution = Eigen::ArrayXd::Constant(10, -1.0);
    EXPECT_FALSE(IsFeasible(individual));

    individual.solution = Eigen::ArrayXd::Constant(10, 2.0);
    EXPECT_FALSE(IsFeasible(individual));
}

TEST_F(DTLZ1Test, EvaluateConstraints) {
    Individuald individual{Eigen::ArrayXd::Zero(10)};
    auto evaluation = EvaluateConstraints(individual);
    for (int i = 0; i < evaluation.size(); ++i) {
        EXPECT_TRUE(evaluation[i]);
    }

    individual.solution = Eigen::ArrayXd::Ones(10);
    evaluation = EvaluateConstraints(individual);
    for (int i = 0; i < evaluation.size(); ++i) {
        EXPECT_TRUE(evaluation[i]);
    }

    individual.solution = Eigen::ArrayXd::Constant(10, 0.5);
    evaluation = EvaluateConstraints(individual);
    for (int i = 0; i < evaluation.size(); ++i) {
        EXPECT_TRUE(evaluation[i]);
    }

    individual.solution = Eigen::ArrayXd::Constant(10, -1.0);
    evaluation = EvaluateConstraints(individual);
    for (int i = 0; i < evaluation.size(); ++i) {
        EXPECT_FALSE(evaluation[i]);
    }

    individual.solution = Eigen::ArrayXd::Constant(10, 2.0);
    evaluation = EvaluateConstraints(individual);
    for (int i = 0; i < evaluation.size(); ++i) {
        EXPECT_FALSE(evaluation[i]);
    }
}

TEST_F(DTLZ2Test, ComputeObjectiveSet) {
    for (int i = 0; i < X.size(); ++i) {
        Individuald individual{
            Eigen::Map<const Eigen::ArrayXd>(X[i].data(), X[i].size())};
        ComputeObjectiveSet(individual);
        for (int j = 0; j < F[i].size(); ++j) {
            EXPECT_DOUBLE_EQ(individual.objectives(j), F[i][j]);
        }
    }
}

TEST_F(DTLZ3Test, ComputeObjectiveSet) {
    for (int i = 0; i < X.size(); ++i) {
        Individuald individual{
            Eigen::Map<const Eigen::ArrayXd>(X[i].data(), X[i].size())};
        ComputeObjectiveSet(individual);
        for (int j = 0; j < F[i].size(); ++j) {
            EXPECT_DOUBLE_EQ(individual.objectives(j), F[i][j]);
        }
    }
}

TEST_F(DTLZ4Test, ComputeObjectiveSet) {
    for (int i = 0; i < X.size(); ++i) {
        Individuald individual{
            Eigen::Map<const Eigen::ArrayXd>(X[i].data(), X[i].size())};
        ComputeObjectiveSet(individual);
        for (int j = 0; j < F[i].size(); ++j) {
            EXPECT_DOUBLE_EQ(individual.objectives(j), F[i][j]);
        }
    }
}

TEST_F(DTLZ5Test, ComputeObjectiveSet) {
    for (int i = 0; i < X.size(); ++i) {
        Individuald individual{
            Eigen::Map<const Eigen::ArrayXd>(X[i].data(), X[i].size())};
        ComputeObjectiveSet(individual);
        for (int j = 0; j < F[i].size(); ++j) {
            EXPECT_DOUBLE_EQ(individual.objectives(j), F[i][j]);
        }
    }
}

TEST_F(DTLZ6Test, ComputeObjectiveSet) {
    for (int i = 0; i < X.size(); ++i) {
        Individuald individual{
            Eigen::Map<const Eigen::ArrayXd>(X[i].data(), X[i].size())};
        ComputeObjectiveSet(individual);
        for (int j = 0; j < F[i].size(); ++j) {
            EXPECT_DOUBLE_EQ(individual.objectives(j), F[i][j]);
        }
    }
}

TEST_F(DTLZ7Test, ComputeObjectiveSet) {
    for (int i = 0; i < X.size(); ++i) {
        Individuald individual{
            Eigen::Map<const Eigen::ArrayXd>(X[i].data(), X[i].size())};
        ComputeObjectiveSet(individual);
        for (int j = 0; j < F[i].size(); ++j) {
            EXPECT_DOUBLE_EQ(individual.objectives(j), F[i][j]);
        }
    }
}

}  // namespace Eacpp::Test