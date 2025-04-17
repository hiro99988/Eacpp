#pragma once

#include <Eigen/Core>
#include <array>
#include <tuple>
#include <vector>

#include "Individual.h"
#include "Problems/IProblem.h"

namespace Eacpp {

class DTLZBase : public IProblem<double> {
   public:
    virtual ~DTLZBase() {}

    int DecisionVariablesNum() const override;
    int ObjectivesNum() const override;
    const std::vector<std::pair<double, double>>& VariableBounds()
        const override;
    void ComputeObjectiveSet(Individuald& individual) const override;
    bool IsFeasible(const Individuald& individual) const override;
    std::vector<bool> EvaluateConstraints(
        const Individuald& individual) const override;

   protected:
    DTLZBase(int decisionVariablesNum, int objectivesNum)
        : _decisionVariablesNum(decisionVariablesNum),
          _objectivesNum(objectivesNum) {}

    virtual double G(const Eigen::ArrayXd& XM) const = 0;
    virtual Eigen::ArrayXd Objectives(const Eigen::ArrayXd& X_,
                                      double g) const = 0;

    double G13(const Eigen::ArrayXd& XM) const;
    double G245(const Eigen::ArrayXd& XM) const;
    Eigen::ArrayXd Objectives234(const Eigen::ArrayXd& X_, double g,
                                 double alpha = 1.0) const;
    Eigen::ArrayXd Objectives56(const Eigen::ArrayXd& X_, double g) const;
    Eigen::ArrayXd theta56(const Eigen::ArrayXd& X_, double g) const;

   private:
    int _decisionVariablesNum;
    int _objectivesNum;
    const std::vector<std::pair<double, double>> _variableBounds = {{0.0, 1.0}};
};

class DTLZ1 : public DTLZBase {
   public:
    DTLZ1(int decisionVariablesNum, int objectivesNum)
        : DTLZBase(decisionVariablesNum, objectivesNum) {}

   private:
    double G(const Eigen::ArrayXd& XM) const override {
        return G13(XM);
    }

    Eigen::ArrayXd Objectives(const Eigen::ArrayXd& X_,
                              double g) const override;

#ifdef _TEST_
   public:
    friend class DTLZ1Test;
#endif
};

class DTLZ2 : public DTLZBase {
   public:
    DTLZ2(int decisionVariablesNum, int objectivesNum)
        : DTLZBase(decisionVariablesNum, objectivesNum) {}

   private:
    double G(const Eigen::ArrayXd& XM) const override {
        return G245(XM);
    }

    Eigen::ArrayXd Objectives(const Eigen::ArrayXd& X_,
                              double g) const override {
        return Objectives234(X_, g);
    }

#ifdef _TEST_
   public:
    friend class DTLZ2Test;
#endif
};

class DTLZ3 : public DTLZBase {
   public:
    DTLZ3(int decisionVariablesNum, int objectivesNum)
        : DTLZBase(decisionVariablesNum, objectivesNum) {}

   private:
    double G(const Eigen::ArrayXd& XM) const override {
        return G13(XM);
    }

    Eigen::ArrayXd Objectives(const Eigen::ArrayXd& X_,
                              double g) const override {
        return Objectives234(X_, g);
    }

#ifdef _TEST_
   public:
    friend class DTLZ3Test;
#endif
};

class DTLZ4 : public DTLZBase {
   public:
    DTLZ4(int decisionVariablesNum, int objectivesNum)
        : DTLZBase(decisionVariablesNum, objectivesNum) {}
    DTLZ4(int decisionVariablesNum, int objectivesNum, int alpha)
        : DTLZBase(decisionVariablesNum, objectivesNum), _alpha(alpha) {}

   private:
    int _alpha = 100;

    double G(const Eigen::ArrayXd& XM) const override {
        return G245(XM);
    }

    Eigen::ArrayXd Objectives(const Eigen::ArrayXd& X_,
                              double g) const override {
        return Objectives234(X_, g, _alpha);
    }

#ifdef _TEST_
   public:
    friend class DTLZ4Test;
#endif
};

class DTLZ5 : public DTLZBase {
   public:
    DTLZ5(int decisionVariablesNum, int objectivesNum)
        : DTLZBase(decisionVariablesNum, objectivesNum) {}

   private:
    double G(const Eigen::ArrayXd& XM) const override {
        return G245(XM);
    }

    Eigen::ArrayXd Objectives(const Eigen::ArrayXd& X_,
                              double g) const override {
        return Objectives56(X_, g);
    }

#ifdef _TEST_
   public:
    friend class DTLZ5Test;
#endif
};

class DTLZ6 : public DTLZBase {
   public:
    DTLZ6(int decisionVariablesNum, int objectivesNum)
        : DTLZBase(decisionVariablesNum, objectivesNum) {}

   private:
    double G(const Eigen::ArrayXd& XM) const override;

    Eigen::ArrayXd Objectives(const Eigen::ArrayXd& X_,
                              double g) const override {
        return Objectives56(X_, g);
    }

#ifdef _TEST_
   public:
    friend class DTLZ6Test;
#endif
};

class DTLZ7 : public DTLZBase {
   public:
    DTLZ7(int decisionVariablesNum, int objectivesNum)
        : DTLZBase(decisionVariablesNum, objectivesNum) {}

   private:
    double G(const Eigen::ArrayXd& XM) const override;
    Eigen::ArrayXd Objectives(const Eigen::ArrayXd& X_,
                              double g) const override;
    double H(const Eigen::ArrayXd& F_, double g) const;

#ifdef _TEST_
   public:
    friend class DTLZ7Test;
#endif
};

}  // namespace Eacpp