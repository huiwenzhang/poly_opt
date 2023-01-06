#pragma once
#include <sys/time.h>

#include <numeric>
#include <set>

#include "poly_traj_generation/convolution.h"
#include "poly_traj_generation/poly_opt_linear.h"

namespace poly_opt {

template <int _N>
PolyOptimization<_N>::PolyOptimization(size_t dimension)
        : dimension_(dimension),
          derivative_to_optimize_(derivative_order::INVALID),
          n_vertices_(0),
          n_segments_(0),
          n_all_constraints_(0),
          n_fixed_constraints_(0),
          n_free_constraints_(0) {
    fixed_constraints_compact_.resize(dimension_);
    free_constraints_compact_.resize(dimension_);
}

template <int _N>
bool PolyOptimization<_N>::setupFromVertices(const Vertex::Vector& vertices,
                                             const std::vector<double>& times,
                                             int derivative_to_optimize) {
    assert(derivative_to_optimize >= 0 &&
           derivative_to_optimize <= kHighestDerivativeToOptimize);
    derivative_to_optimize_ = derivative_to_optimize;
    vertices_ = vertices;
    // segment_times_ = times;

    n_vertices_ = vertices_.size();
    n_segments_ = n_vertices_ - 1;

    segments_.resize(n_segments_, Segment(N, dimension_));

    assert(n_vertices_ == times.size() + 1);

    inverse_mapping_matrices_.resize(n_segments_);
    cost_matrices_.resize(n_segments_);

    for (size_t vertex_idx = 0; vertex_idx < n_vertices_; vertex_idx++) {
        Vertex& vertex = vertices_[vertex_idx];

        bool vertex_valid = true;
        Vertex tmp(dimension_);  // tmp vertex removes invalid constraints
        for (Vertex::Constraints::const_iterator it = vertex.cBegin();
             it != vertex.cEnd();
             ++it) {
            if (it->first > kHighestDerivativeToOptimize) {
                vertex_valid = false;
                // mwarn("invalid vertex %d, optimize order %d exceeds highest order %d",
                //       vertex_idx,
                //       it->first,
                //       kHighestDerivativeToOptimize);
            } else {
                tmp.addConstraint(it->first, it->second);
            }

            if (!vertex_valid) { vertex = tmp; }  // 滤除不合理的约束
        }
    }
    updateSegmentTimes(times);
    setupConstraintReorderMatrix();
    return true;
    // printReorderingMatrix(std::cout);
}

// 根据每个segment的时间计算A，Q
template <int _N>
void PolyOptimization<_N>::updateSegmentTimes(const std::vector<double>& segment_times) {
    const size_t n_times = segment_times.size();
    assert(n_times == n_segments_);

    segment_times_ = segment_times;
    for (int i = 0; i < n_segments_; i++) {
        const double segment_time = segment_times_[i];
        assert(segment_time > 0);
        computeQuadraticCostJacobian(
                derivative_to_optimize_, segment_time, &cost_matrices_[i]);
        SquareMatrix A;
        setupMappingMatrix(segment_time, &A);
        invertMappingMatrix(A, &inverse_mapping_matrices_[i]);
        // std::cout << cost_matrices_[i] << std::endl;
        // std::cout << A << std::endl;
        // std::cout << inverse_mapping_matrices_[i] * A << std::endl;
    }
}

// 计算Q
template <int _N>
void PolyOptimization<_N>::computeQuadraticCostJacobian(int derivative,
                                                        double t,
                                                        SquareMatrix* cost_jacobian) {
    assert(derivative < N);
    cost_jacobian->setZero();
    for (int col = 0; col < N - derivative; col++) {
        for (int row = 0; row < N - derivative; row++) {
            // N -1 - row/col = index, index - derivative = single power, + 1-->integral
            double power = (N - 1 - derivative) * 2 + 1 - row - col;
            (*cost_jacobian)(N - 1 - row, N - 1 - col) =
                    Polynomial::base_coefficients_(derivative, N - 1 - row) *
                    Polynomial::base_coefficients_(derivative, N - 1 - col) *
                    pow(t, power) * 2.0 / power;
        }
    }
}

// 计算A
template <int _N>
void PolyOptimization<_N>::setupMappingMatrix(double segment_time, SquareMatrix* A) {
    // 约束的导数都一样，时间分别是0和segment_time，因此第i行和N/2+i行的系数一样
    for (int i = 0; i < N / 2; i++) {
        A->row(i) = Polynomial::baseCoeffsWithTime(N, i, 0.0);
        A->row(N / 2 + i) = Polynomial::baseCoeffsWithTime(N, i, segment_time);
    }
}

template <int _N>
void PolyOptimization<_N>::invertMappingMatrix(const SquareMatrix& mapping_matrix,
                                               SquareMatrix* inverse_mapping_matrix) {
    // The mapping matrix has the following structure:
    // [ x 0 0 0 0 0 ]
    // [ 0 x 0 0 0 0 ]
    // [ 0 0 x 0 0 0 ]
    // [ x x x x x x ]
    // [ 0 x x x x x ]
    // [ 0 0 x x x x ]
    // ==>
    // [ A_diag B=0 ]
    // [ C      D   ]
    // We make use of the Schur-complement, so the inverse is:
    // [ inv(A_diag)               0      ]
    // [ -inv(D) * C * inv(A_diag) inv(D) ]
    const int half_n = N / 2;
    const Eigen::Matrix<double, half_n, 1> A_diag =
            mapping_matrix.template block<half_n, half_n>(0, 0).diagonal();
    const Eigen::Matrix<double, half_n, half_n> A_inv =
            A_diag.cwiseInverse().asDiagonal();

    const Eigen::Matrix<double, half_n, half_n> C =
            mapping_matrix.template block<half_n, half_n>(half_n, 0);

    const Eigen::Matrix<double, half_n, half_n> D_inv =
            mapping_matrix.template block<half_n, half_n>(half_n, half_n).inverse();

    inverse_mapping_matrix->template block<half_n, half_n>(0, 0) = A_inv;
    inverse_mapping_matrix->template block<half_n, half_n>(0, half_n).setZero();
    inverse_mapping_matrix->template block<half_n, half_n>(half_n, 0) =
            -D_inv * C * A_inv;
    inverse_mapping_matrix->template block<half_n, half_n>(half_n, half_n) = D_inv;
}

// 计算目标函数值
template <int _N>
double PolyOptimization<_N>::computeCost() const {
    assert(n_segments_ == segments_.size() && n_segments_ == cost_matrices_.size());
    double cost = 0.0;
    for (size_t sidx = 0; sidx < n_segments_; sidx++) {
        const SquareMatrix& Q = cost_matrices_[sidx];
        const Segment& segment = segments_[sidx];
        for (size_t dimension_idx = 0; dimension_idx < dimension_; dimension_idx++) {
            const Eigen::VectorXd c =
                    segment[dimension_idx].getCoefficients(derivative_order::POSITION);
            cost += c.transpose() * Q * c;
        }
    }
    return 0.5 * cost;
}

// 对mapping matrix进行整理
template <int _N>
void PolyOptimization<_N>::setupConstraintReorderMatrix() {
    typedef Eigen::Triplet<double> Triplet;  // hold (i, j, value) in sparse matrix
    std::vector<Triplet> reorder_list;

    const size_t n_vertices = vertices_.size();

    std::vector<Constraint> all_constraints;
    std::set<Constraint> fixed_constraints;
    std::set<Constraint> free_constraints;

    all_constraints.reserve((n_vertices - 1) * N);

    // Extract constraints and sort them to fixed and free. For the start and
    // end Vertex, we need to do this once, while we need to do it twice for the
    // other vertices, since constraints are shared and enforce continuity.
    // 这里并没有将连续性条件显示带入，而是构造了更多的约束
    for (size_t vertex_idx = 0; vertex_idx < n_vertices; ++vertex_idx) {
        const Vertex& vertex = vertices_[vertex_idx];

        int n_constraint_occurence = 2;
        if (vertex_idx == 0 || vertex_idx == n_segments_) { n_constraint_occurence = 1; }

        for (int co = 0; co < n_constraint_occurence; ++co) {
            for (size_t cons_idx = 0; cons_idx < N / 2; cons_idx++) {
                Constraint constraint;
                constraint.vertex_idx = vertex_idx;
                constraint.constraint_idx = cons_idx;
                bool has_cons = vertex.getConstraint(cons_idx, &(constraint.value));
                if (has_cons) {
                    // std::cout << constraint.value << std::endl;
                    all_constraints.push_back(constraint);
                    fixed_constraints.insert(constraint);  // 有设定值的约束都是固定约束
                } else {
                    // 为什么free constraints可以给任意值
                    constraint.value = Vertex::ConstraintValue::Constant(dimension_, 0);
                    all_constraints.push_back(constraint);
                    free_constraints.insert(constraint);
                }
            }
        }
    }

    // non-compact constraints
    n_all_constraints_ = all_constraints.size();
    n_fixed_constraints_ = fixed_constraints.size();
    n_free_constraints_ = free_constraints.size();

    // minfo("constraint size, all: %d, fixed: %d, free: %d",
    //       n_all_constraints_,
    //       n_fixed_constraints_,
    //       n_free_constraints_);

    reorder_list.reserve(n_all_constraints_);
    // 这里的reordering矩阵实际完成了2个功能，第一：将fixed_cons和free_cons分开，
    // 第二：把一个约束映射到了2个，实现了矩阵M的功能
    constraint_reordering_ = Eigen::SparseMatrix<double>(
            n_all_constraints_, n_free_constraints_ + n_fixed_constraints_);

    for (Eigen::VectorXd& df : fixed_constraints_compact_) {
        df.resize(n_fixed_constraints_, Eigen::NoChange);
    }

    int row = 0;
    int col = 0;
    for (const Constraint& cons : all_constraints) {
        for (const Constraint& cf : fixed_constraints) {
            if (cons == cf) {
                reorder_list.emplace_back(Triplet(row, col, 1.0));
                for (size_t d = 0; d < dimension_; d++) {
                    Eigen::VectorXd& df = fixed_constraints_compact_[d];
                    const Eigen::VectorXd cons_all_dimensions = cf.value;
                    df[col] = cons_all_dimensions[d];
                }
            }
            ++col;
        }
        // 这里先找fixed，所以free的col肯定在fxied的后面
        for (const Constraint& cp : free_constraints) {
            if (cons == cp) reorder_list.emplace_back(Triplet(row, col, 1.0));
            ++col;  // 在fixed的基础上递增
        }
        col = 0;
        ++row;
    }
    // std::cout << "fixed constraints\n" << fixed_constraints_compact_[0] << std::endl;
    constraint_reordering_.setFromTriplets(reorder_list.begin(), reorder_list.end());
}

template <int _N>
void PolyOptimization<_N>::updateSegmentsFromCompactConstraints() {
    const size_t n_all_constraints = n_fixed_constraints_ + n_free_constraints_;

    // for each dimension
    for (size_t idx = 0; idx < dimension_; ++idx) {
        const Eigen::VectorXd& df = fixed_constraints_compact_[idx];
        const Eigen::VectorXd& dp = free_constraints_compact_[idx];

        Eigen::VectorXd d_all(n_all_constraints);
        d_all << df, dp;

        // for each segment
        for (size_t i = 0; i < n_segments_; ++i) {
            const Eigen::Matrix<double, N, 1> new_d =
                    constraint_reordering_.block(i * N, 0, N, n_all_constraints) * d_all;
            const Eigen::Matrix<double, N, 1> coeffs =
                    inverse_mapping_matrices_[i] * new_d;  // 更新后的系数
            Segment& segment = segments_[i];
            segment.setTime(segment_times_[i]);
            segment[idx] = Polynomial(N, coeffs);
        }
    }
}

template <int _N>
void PolyOptimization<_N>::constructR(Eigen::SparseMatrix<double>* R) const {
    assert(R != nullptr);
    typedef Eigen::Triplet<double> Triplet;
    std::vector<Triplet> cost_unconstrained_triplets;
    cost_unconstrained_triplets.reserve(N * N * n_segments_);

    for (size_t i = 0; i < n_segments_; ++i) {
        const SquareMatrix& Ai = inverse_mapping_matrices_[i];
        const SquareMatrix& Q = cost_matrices_[i];
        const SquareMatrix H = Ai.transpose() * Q * Ai;
        const int start_pos = i * N;
        for (int row = 0; row < N; ++row) {
            for (int col = 0; col < N; ++col) {
                cost_unconstrained_triplets.emplace_back(
                        Triplet(start_pos + row, start_pos + col, H(row, col)));
            }
        }
    }
    Eigen::SparseMatrix<double> cost_unconstrained(N * n_segments_, N * n_segments_);
    // 将矩阵拼接起来
    // [H0 ----------]
    // [---H1 -------]
    // [------H2-----]
    cost_unconstrained.setFromTriplets(cost_unconstrained_triplets.begin(),
                                       cost_unconstrained_triplets.end());

    // [1]: R = C^T * H * C. C: constraint_reodering_ ; H: cost_unconstrained,
    // assembled from the block-H above.
    *R = constraint_reordering_.transpose() * cost_unconstrained * constraint_reordering_;
}

template <int _N>
bool PolyOptimization<_N>::solveLinear() {
    timeval t_start, t_end;
    gettimeofday(&t_start, NULL);
    assert(derivative_to_optimize_ >= 0 &&
           derivative_to_optimize_ <= kHighestDerivativeToOptimize);
    // Catch the fully constrained case:
    if (n_free_constraints_ == 0) {
        // mwarn("No free constraints set in the vertices. Polynomial can "
        //       "not be optimized. Outputting fully constrained polynomial.");
        updateSegmentsFromCompactConstraints();
        return true;
    }

    // TODO(acmarkus): figure out if sparse becomes less efficient for small
    // problems, and switch back to dense in case.

    // Compute cost matrix for the unconstrained optimization problem.
    // Block-wise H = A^{-T}QA^{-1} according to [1]
    Eigen::SparseMatrix<double> R;
    constructR(&R);

    // Extract block matrices and prepare solver.
    Eigen::SparseMatrix<double> Rpf =
            R.block(n_fixed_constraints_, 0, n_free_constraints_, n_fixed_constraints_);
    Eigen::SparseMatrix<double> Rpp = R.block(n_fixed_constraints_,
                                              n_fixed_constraints_,
                                              n_free_constraints_,
                                              n_free_constraints_);
    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
    solver.compute(Rpp);

    // Compute dp_opt for every dimension.
    // 不同维度的A，C相同，因此R也是共用的
    for (size_t dimension_idx = 0; dimension_idx < dimension_; ++dimension_idx) {
        Eigen::VectorXd df =
                -Rpf * fixed_constraints_compact_[dimension_idx];  // Rpf = Rfp^T
        // std::cout << "df: " << fixed_constraints_compact_[dimension_idx].transpose() << std::endl;
        // solver AX = B by QR decomposition
        free_constraints_compact_[dimension_idx] =
                solver.solve(df);  // dp = -Rpp^-1 * Rpf * df
        // std::cout << "dp: " << free_constraints_compact_[dimension_idx].transpose() << std::endl;
    }

    updateSegmentsFromCompactConstraints();  // 更新优化后的系数

    gettimeofday(&t_end, NULL);
    // minfo("Linear solver takes %f secs", (t_end.tv_usec - t_start.tv_usec) / 1.0e6);
    return true;
}

template <int _N>
void PolyOptimization<_N>::printReorderingMatrix(std::ostream& stream) const {
    stream << "Mapping matrix:\n" << constraint_reordering_ << std::endl;
}

template <int _N>
void PolyOptimization<_N>::setFreeConstraints(
        const std::vector<Eigen::VectorXd>& free_constraints) {
    assert(free_constraints.size() == dimension_);
    for (const Eigen::VectorXd& v : free_constraints) {
        assert(v.size() == n_free_constraints_);
    }

    free_constraints_compact_ = free_constraints;
}

template <int _N>
void PolyOptimization<_N>::getAInverse(Eigen::MatrixXd* A_inv) const {
    assert(A_inv != nullptr);
    A_inv->resize(N * n_segments_, N * n_segments_);
    A_inv->setZero();

    for (size_t i = 0; i < n_segments_; i++) {
        (*A_inv).block<N, N>(N * i, N * i) = inverse_mapping_matrices_[i];
    }
}

template <int _N>
void PolyOptimization<_N>::getM(Eigen::MatrixXd* M) const {
    assert(M != nullptr);
    *M = constraint_reordering_;
}

template <int _N>
void PolyOptimization<_N>::getR(Eigen::MatrixXd* R) const {
    assert(R != nullptr);

    Eigen::SparseMatrix<double> R_sparse;
    constructR(&R_sparse);

    *R = R_sparse;
}

template <int _N>
void PolyOptimization<_N>::getA(Eigen::MatrixXd* A) const {
    assert(A != nullptr);
    A->resize(N * n_segments_, N * n_segments_);
    A->setZero();

    // Create a mapping matrix per segment and append them together.
    for (size_t i = 0; i < n_segments_; ++i) {
        const double segment_time = segment_times_[i];
        assert(segment_time > 0);

        SquareMatrix A_segment;
        setupMappingMatrix(segment_time, &A_segment);

        (*A).block<N, N>(N * i, N * i) = A_segment;
    }
}

template <int _N>
void PolyOptimization<_N>::getMpinv(Eigen::MatrixXd* M_pinv) const {
    assert(M_pinv != nullptr);

    // Pseudoinverse implementation by @SebastianInd.
    *M_pinv = constraint_reordering_.transpose();
    for (int M_row = 0; M_row < M_pinv->rows(); M_row++) {
        M_pinv->row(M_row) = M_pinv->row(M_row) / M_pinv->row(M_row).sum();
    }
}

//*******************extremum computation*************
template <int _N>
template <int Derivative>
bool PolyOptimization<_N>::computeSegmentMaximumMagnitudeCandidates(
        const Segment& segment,
        double t_start,
        double t_stop,
        std::vector<double>* candidates) {
    return computeSegmentMaximumMagnitudeCandidates(
            Derivative, segment, t_start, t_stop, candidates);
}

template <int _N>
bool PolyOptimization<_N>::computeSegmentMaximumMagnitudeCandidates(
        int derivative,
        const Segment& segment,
        double t_start,
        double t_stop,
        std::vector<double>* candidates) {
    assert(candidates != nullptr);
    assert(N - derivative - 1 > 0);

    // Use the implementation of this in the segment (template-free) as it's
    // actually faster.
    std::vector<int> dimensions(segment.D());
    std::iota(dimensions.begin(), dimensions.end(), 0);
    return segment.computeMinMaxMagnitudeCandidateTimes(
            derivative, t_start, t_stop, dimensions, candidates);
}

template <int _N>
template <int Derivative>
void PolyOptimization<_N>::computeSegmentMaximumMagnitudeCandidatesBySampling(
        const Segment& segment,
        double t_start,
        double t_stop,
        double dt,
        std::vector<double>* candidates) {
    assert(candidates != nullptr);
    // Start is candidate.
    candidates->push_back(t_start);

    // Determine initial direction from t_start to t_start + dt.
    auto t_old = t_start + dt;
    auto value_new = segment.evaluate(t_old, Derivative);
    auto value_old = segment.evaluate(t_start, Derivative);
    auto direction = value_new.norm() - value_old.norm();

    // Continue with direction from t_start + dt to t_start + 2 dt until t_stop.
    bool last_sample = false;
    for (double t = t_start + dt + dt; t <= t_stop; t += dt) {
        // Update direction.
        value_old = value_new;
        value_new = segment.evaluate(t, Derivative);
        auto direction_new = value_new.norm() - value_old.norm();

        if (std::signbit(direction) != std::signbit(direction_new)) {
            auto value_deriv = segment.evaluate(t_old, Derivative + 1);
            if (value_deriv.norm() < 1e-2) {
                candidates->push_back(t_old);  // extremum was at last dt
            }
        }

        direction = direction_new;
        t_old = t;

        // Check last sample before t_stop.
        if ((t + dt) > t_stop && !last_sample) {
            t = t_stop - dt;
            last_sample = true;
        }
    }

    // End is candidates.
    if (candidates->back() != t_stop) { candidates->push_back(t_stop); }
}

template <int _N>
template <int Derivative>
Extremum PolyOptimization<_N>::computeMaximumOfMagnitude(
        std::vector<Extremum>* candidates) const {
    return computeMaximumOfMagnitude(Derivative, candidates);
}

template <int _N>
Extremum PolyOptimization<_N>::computeMaximumOfMagnitude(
        int derivative,
        std::vector<Extremum>* candidates) const {
    if (candidates != nullptr) candidates->clear();

    int segment_idx = 0;
    Extremum extremum;
    for (const Segment& s : segments_) {
        std::vector<double> extrema_times;
        extrema_times.reserve(N - 1);
        // Add the beginning as well. Call below appends its extrema.
        extrema_times.push_back(0.0);
        computeSegmentMaximumMagnitudeCandidates(
                derivative, s, 0.0, s.getTime(), &extrema_times);

        for (double t : extrema_times) {
            const Extremum candidate(t, s.evaluate(t, derivative).norm(), segment_idx);
            if (extremum < candidate) extremum = candidate;
            if (candidates != nullptr) candidates->emplace_back(candidate);
        }
        ++segment_idx;
    }
    // Check last time at last segment.
    const Extremum candidate(
            segments_.back().getTime(),
            segments_.back().evaluate(segments_.back().getTime(), derivative).norm(),
            n_segments_ - 1);
    if (extremum < candidate) extremum = candidate;
    if (candidates != nullptr) candidates->emplace_back(candidate);

    return extremum;
}

}  // namespace poly_opt