#pragma once
#include <nlopt.hpp>
#include <memory>
#include "poly_traj_generation/poly_opt_linear.h"

namespace poly_opt {
constexpr double kOptimizationTimeLowerBound = 0.1;

// 非线性优化参数
struct NonlinearOptimizationParameters {
    double f_abs = -1;
    double f_rel = 0.05;
    double x_rel = -1;
    double x_abs = -1;
    double initial_stepsize_rel = 0.1;
    double equality_constraint_tolerance = 1.0e-3;
    double inequality_constraint_tolerance = 0.1;
    int max_iters = 3000;
    double time_penalty = 500.0;
    nlopt::algorithm algorithm = nlopt::LN_BOBYQA;
    int random_seed = 0;
    bool use_soft_constraints = true;
    double soft_constraint_weight = 100.0;

    enum TimeAllocMethod {
        kSquaredTime,
        kRichterTime,
        kMellingerOuterLoop,
        kSquaredTimeAndConstraints,
        kRichterTimeAndConstraints,
        kUnknown
    } time_alloc_method = kSquaredTimeAndConstraints;

    bool print_debug_info = false;
    bool print_debug_info_time_allocation = false;
};

struct OptimizationInfo {
    int n_iterations = 0;
    int stopping_reason = nlopt::FAILURE;
    double cost_trajectory = 0.0;
    double cost_time = 0.0;
    double cost_soft_constraints = 0.0;
    double optimization_time = 0.0;
    std::map<int, Extremum> maxima;
};

std::ostream& operator<<(std::ostream& stream, const OptimizationInfo& val);

template <int _N = 8>
class PolynomialOptimizationNonLinear {
    static_assert(_N % 2 == 0, "The number of coefficients has to be even.");

public:
    enum { N = _N };
    PolynomialOptimizationNonLinear(size_t dimension,
                                    const NonlinearOptimizationParameters& parameters);
    bool setupFromVertices(const Vertex::Vector& vertices,
                           const std::vector<double>& segment_times,
                           int derivative_to_optimize =
                                   PolyOptimization<N>::kHighestDerivativeToOptimize);

    bool addMaximumMagnitudeConstraint(int derivative_order, double maximum_value);
    bool solveLinear();
    int optimize();
    void getTrajectory(Trajectory* trajectory) const {
        poly_opt_.getTrajectory(trajectory);
    }

    const PolyOptimization<N>& getPolyLinearOptRef() { return poly_opt_; }

    OptimizationInfo getOptInfo() const { return opt_info_; }

    double getCost() const;
    double getTotalCostWithSoftConstraints() const;
    void scaleSegmentTimesWithViolation();

private:
    struct ConstraintData {
        PolynomialOptimizationNonLinear<N>* this_object;
        int derivative;
        double value;
    };

    // nlopt objective func.
    // The function f should be of the form:
    // double f(const std::vector`<double>` &x, std::vector`<double>` &grad, void*
    // f_data);
    // ref: https://nlopt.readthedocs.io/en/latest/NLopt_C-plus-plus_Reference/
    static double objectiveFunctionTime(const std::vector<double>& segment_times,
                                        std::vector<double>& gradient,
                                        void* data);
    static double objectiveFunctionTimeMellingerOuterLoop(
            const std::vector<double>& segment_times,
            std::vector<double>& gradient,
            void* data);
    static double objectiveFunctionTimeAndConstraints(
            const std::vector<double>& optimization_variables,
            std::vector<double>& gradient,
            void* data);
    static double evaluateMaximumMagnitudeConstraint(
            const std::vector<double>& optimization_variables,
            std::vector<double>& gradient,
            void* data);

    // Does the actual optimization work for the time-only version.
    int optimizeTime();
    int optimizeTimeMellingerOuterLoop();

    // Does the actual optimization work for the full optimization version.
    int optimizeTimeAndFreeConstraints();

    double evaluateMaximumMagnitudeAsSoftConstraint(
            const std::vector<std::shared_ptr<ConstraintData>>& inequality_constraints,
            double weight,
            double maximum_cost = 1.0e12) const;

    // Set lower and upper bounds on the optimization parameters
    void setFreeEndpointDerivativeHardConstraints(const Vertex::Vector& vertices,
                                                  std::vector<double>* lower_bounds,
                                                  std::vector<double>* upper_bounds);

    // Computes the gradients by doing forward difference!
    double getCostAndGradientMellinger(std::vector<double>* gradients);

    // Computes the total trajectory time.
    static double computeTotalTrajectoryTime(const std::vector<double>& segment_times);

    std::shared_ptr<nlopt::opt> nlopt_;
    PolyOptimization<N> poly_opt_;
    NonlinearOptimizationParameters opt_params_;
    std::vector<std::shared_ptr<ConstraintData>> inequality_constraints_;
    OptimizationInfo opt_info_;
};
}  // namespace poly_opt

namespace nlopt {
std::string returnValueToString(int value);
}

#include "poly_traj_generation/impl/poly_opt_nonlinear_impl.h"