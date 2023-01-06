#pragma once

#include <Eigen/Sparse>
#include <tuple>

#include "poly_traj_generation/extremum.h"
#include "poly_traj_generation/motion_defines.h"
#include "poly_traj_generation/polynomial.h"
#include "poly_traj_generation/segment.h"
#include "poly_traj_generation/trajectory.h"
#include "poly_traj_generation/vertex.h"

namespace poly_opt {
// _N = Number of coefficients of the underlying polynomials.
// Polynomial coefficients are stored with increasing powers,
// i.e. c_0 + c_1*t ... c_{N-1} * t^{N-1}.
template <int _N = 8>
class PolyOptimization {
    static_assert(_N % 2 == 0, "The number of coeffs has to be even.");

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    enum { N = _N };
    static constexpr int kHighestDerivativeToOptimize = N / 2 - 1;
    typedef Eigen::Matrix<double, N, N> SquareMatrix;
    typedef std::vector<SquareMatrix, Eigen::aligned_allocator<SquareMatrix> >
            SquareMatrixVector;

    PolyOptimization(size_t dimension);

    bool setupFromVertices(const Vertex::Vector& vertices,
                           const std::vector<double>& segment_times,
                           int derivative_to_optimize = kHighestDerivativeToOptimize);

    bool setupFromPositions(const std::vector<double>& positions,
                            const std::vector<double>& times);

    static void invertMappingMatrix(const SquareMatrix& mapping_matrix,
                                    SquareMatrix* inverse_mapping_matrix);

    static void setupMappingMatrix(double segment_time, SquareMatrix* A);

    double computeCost() const;

    // Updates the segment times. The number of times has to be equal to
    // the number of vertices that was initially passed during the problem setup.
    // This recomputes all cost- and inverse mapping block-matrices and is meant
    // to be called during non-linear optimization procedures.
    void updateSegmentTimes(const std::vector<double>& segment_times);

    void getTrajectory(Trajectory* trajectory) const {
        assert(trajectory != nullptr);
        trajectory->setSegments(segments_);
    }

    bool solveLinear();

    // Computes the candidates for the maximum magnitude of a single
    // segment in the specified derivative.
    // In the 1D case, it simply returns the roots of the derivative of the
    // segment-polynomial.
    // For higher dimensions, e.g. 3D, we need to find the extrema of
    // \sqrt{x(t)^2 + y(t)^2 + z(t)^2}
    // where x, y, z are polynomials describing the position or the derivative,
    // specified by Derivative.
    // Taking the derivative yields  2 x \dot{x} + 2 y \dot{y} + 2 z \dot{z},
    // which needs to be zero at the extrema. The multiplication of two
    // polynomials is a convolution of their coefficients. Re-ordering by their
    // powers and addition yields a polynomial, for which we can compute the
    // roots. Derivative = Derivative of position, in which to find the maxima.
    // Input: segment = Segment to find the maximum.
    // Input: t_start = Only maxima >= t_start are returned. Usually set to 0.
    // Input: t_stop = Only maxima <= t_stop are returned. Usually set to
    // segment time.
    // Output: candidates = Vector containing the candidate times for a maximum.
    // Returns whether the computation succeeded -- false means no candidates
    // were found by Jenkins-Traub.
    template <int Derivative>
    static bool computeSegmentMaximumMagnitudeCandidates(const Segment& segment,
                                                         double t_start,
                                                         double t_stop,
                                                         std::vector<double>* candidates);

    // Template-free version of above:
    static bool computeSegmentMaximumMagnitudeCandidates(int derivative,
                                                         const Segment& segment,
                                                         double t_start,
                                                         double t_stop,
                                                         std::vector<double>* candidates);

    template <int Derivative>
    static void computeSegmentMaximumMagnitudeCandidatesBySampling(
            const Segment& segment,
            double t_start,
            double t_stop,
            double sampling_interval,
            std::vector<double>* candidates);

    template <int Derivative>
    Extremum computeMaximumOfMagnitude(std::vector<Extremum>* candidates) const;

    // Template-free version of above.
    Extremum computeMaximumOfMagnitude(int derivative,
                                       std::vector<Extremum>* candidates) const;
    void getVertices(Vertex::Vector* vertices) const {
        assert(vertices != nullptr);
        *vertices = vertices_;
    }

    // Only for internal use -- always use getTrajectory() instead if you can!
    void getSegments(Segment::Vector* segments) const {
        assert(segments != nullptr);
        *segments = segments_;
    }

    void getSegmentTimes(std::vector<double>* segment_times) const {
        assert(segment_times != nullptr);
        *segment_times = segment_times_;
    }

    void getFreeConstraints(std::vector<Eigen::VectorXd>* free_constraints) const {
        assert(free_constraints != nullptr);
        *free_constraints = free_constraints_compact_;
    }

    void setFreeConstraints(const std::vector<Eigen::VectorXd>& free_constraints);

    void getFixedConstraints(std::vector<Eigen::VectorXd>* fixed_constraints) const {
        assert(fixed_constraints != nullptr);
        *fixed_constraints = fixed_constraints_compact_;
    }

    // Computes the Jacobian of the integral over the squared derivative
    // Output: cost_jacobian = Jacobian matrix to write into.
    // If C is dynamic, the correct size has to be set.
    // Input: t = time of evaluation
    // Input: derivative used to compute the cost, e.g. snap = 4th derivative
    static void computeQuadraticCostJacobian(int derivative,
                                             double t,
                                             SquareMatrix* cost_jacobian);

    size_t getDimension() const { return dimension_; }
    size_t getNumberSegments() const { return n_segments_; }
    size_t getNumberAllConstraints() const { return n_all_constraints_; }
    size_t getNumberFixedConstraints() const { return n_fixed_constraints_; }
    size_t getNumberFreeConstraints() const { return n_free_constraints_; }
    int getDerivativeToOptimize() const { return derivative_to_optimize_; }

    // Accessor functions for internal matrices.
    void getAInverse(Eigen::MatrixXd* A_inv) const;
    void getM(Eigen::MatrixXd* M) const;
    void getR(Eigen::MatrixXd* R) const;
    // Extras not directly used in the standard optimization:
    void getA(Eigen::MatrixXd* A) const;
    void getMpinv(Eigen::MatrixXd* M_pinv) const;  // Pseudo-inverse of M.

    void printReorderingMatrix(std::ostream& stream) const;

private:
    void constructR(Eigen::SparseMatrix<double>* R) const;

    // setup reorder matrix C, same for each dimension
    void setupConstraintReorderMatrix();

    void setupSegmentsFromCompactConstraints();

    // Updates the segments stored internally from the set of compact fixed
    // and free constraints.
    // 通过优化后得到的多项式系数更新segments的表达式
    void updateSegmentsFromCompactConstraints();

    Vertex::Vector vertices_;
    Segment::Vector segments_;
    Eigen::SparseMatrix<double> constraint_reordering_;       // matrix C
    SquareMatrixVector inverse_mapping_matrices_;             // total constriant A^-1
    SquareMatrixVector cost_matrices_;                        // Q in c^T*Qc
    std::vector<Eigen::VectorXd> fixed_constraints_compact_;  // df in compact
    std::vector<Eigen::VectorXd> free_constraints_compact_;   // dp in compact
    std::vector<double> segment_times_;

    size_t dimension_;  // dimension of vertice
    int derivative_to_optimize_;
    size_t n_vertices_;
    size_t n_segments_;

    size_t n_all_constraints_;
    size_t n_fixed_constraints_;
    size_t n_free_constraints_;
};

struct Constraint {
    inline bool operator<(const Constraint& rhs) const {
        if (vertex_idx < rhs.vertex_idx) return true;
        if (rhs.vertex_idx < vertex_idx) return false;

        if (constraint_idx < rhs.constraint_idx) return true;
        if (rhs.constraint_idx < constraint_idx) return false;
        return false;
    }

    inline bool operator==(const Constraint& rhs) const {
        return vertex_idx == rhs.vertex_idx && constraint_idx == rhs.constraint_idx;
    }

    size_t vertex_idx;
    size_t constraint_idx;
    Vertex::ConstraintValue value;
};

}  // namespace poly_opt

#include "poly_traj_generation/impl/poly_opt_linear_impl.h"
