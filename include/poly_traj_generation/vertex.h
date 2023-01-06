#pragma once

#include <Eigen/Core>
#include <map>
#include <vector>

// #include "log_manager.h"
#include "poly_traj_generation/motion_defines.h"
#include "poly_traj_generation/polynomial.h"

namespace poly_opt {
class Vertex {
public:
    typedef std::vector<Vertex> Vector;
    typedef Eigen::VectorXd ConstraintValue;
    typedef std::pair<int, ConstraintValue> Constraint;
    typedef std::map<int, ConstraintValue> Constraints;

    Vertex(size_t dimension) : D_(dimension) {}

    int D() const { return D_; }

    inline void addConstraint(int derivative_order, double value) {
        constraints_[derivative_order] = ConstraintValue::Constant(D_, value);
    }

    void addConstraint(int type, const Eigen::VectorXd& constraint);
    bool removeConstraint(int type);

    void makeStartOrEnd(const Eigen::VectorXd& constraint, int up_to_derivative);
    void makeStartOrEnd(int up_to_derivative);
    // 各个维度设置成一样的值
    void makeStartOrEnd(double value, int up_to_derivative) {
        makeStartOrEnd(Eigen::VectorXd::Constant(D_, value), up_to_derivative);
    }

    bool hasConstraint(int derivative_order) const;

    bool getConstraint(int derivative_order, Eigen::VectorXd* constraint) const;

    // Returns a const iterator to the first constraint.
    typename Constraints::const_iterator cBegin() const { return constraints_.begin(); }

    // Returns a const iterator to the end of the constraints,
    // i.e. one after the last element.
    typename Constraints::const_iterator cEnd() const { return constraints_.end(); }

    // Returns the number of constraints.
    size_t getNumberOfConstraints() const { return constraints_.size(); }

    // Checks if both lhs and rhs are equal up to tol in case of double values.
    bool isEqualTol(const Vertex& rhs, double tol) const;

    // Get subdimension vertex.
    bool getSubdimension(const std::vector<size_t>& subdimensions,
                         int max_derivative_order,
                         Vertex* subvertex) const;

private:
    int D_;
    Constraints constraints_;
};

std::ostream& operator<<(std::ostream& stream, const Vertex& v);

std::ostream& operator<<(std::ostream& stream, const std::vector<Vertex>& vertices);

// Makes a rough estimate based on v_max and a_max about the time
// required to get from one vertex to the next. Uses the current preferred
// method.
std::vector<double> estimateSegmentTimes(const Vertex::Vector& vertices,
                                         double v_max,
                                         double a_max);

// Calculate the velocity assuming instantaneous constant acceleration a_max
// and straight line rest-to-rest trajectories.
// The time_factor \in [1..Inf] increases the allocated time making the segments
// slower and thus feasibility more likely. This method does not take into
// account the start and goal velocity and acceleration.
std::vector<double> estimateSegmentTimesVelocityRamp(const Vertex::Vector& vertices,
                                                     double v_max,
                                                     double a_max,
                                                     double time_factor = 1.0);
std::vector<double> estimateSegmentOrientationTimesRamp(
        const std::vector<Eigen::Quaterniond>& vertices,
        double v_max,
        double a_max,
        double time_factor = 1.0);

// Makes a rough estimate based on v_max and a_max about the time
// required to get from one vertex to the next.
// t_est = 2 * distance/v_max * (1 + magic_fabian_constant * v_max/a_max * exp(-
// distance/v_max * 2);
// magic_fabian_constant was determined to 6.5 in a student project ...
std::vector<double> estimateSegmentTimesNfabian(const Vertex::Vector& vertices,
                                                double v_max,
                                                double a_max,
                                                double magic_fabian_constant = 6.5);

double computeTimeVelocityRamp(const Eigen::VectorXd& start,
                               const Eigen::VectorXd& goal,
                               double v_max,
                               double a_max);
// compute travel time for two orientations, av--angular velocity, aa-angular acceleration
double computeTimeVelocityRamp(const Eigen::Quaterniond& start,
                               const Eigen::Quaterniond& goal,
                               double av_max,
                               double aa_max);

inline int getHighestDerivativeFromN(int N) { return N / 2 - 1; }

// Creates random vertices for position within minimum_position and
// maximum_position.
// Vertices at the beginning and end have only fixed constraints with their
// derivative set to zero, while  all vertices in between have position as fixed
// constraint and the derivatives are left free.
// Input: maximum_derivative = The maximum derivative to be set to zero for
// beginning and end.
// Input: n_segments = Number of segments of the resulting trajectory. Number
// of vertices is n_segments + 1.
// Input: minimum_position = Minimum position of the space to sample.
// Input: maximum_position = Maximum position of the space to sample.
// Input: seed = Initial seed for random number generation.
// Output: return = Vector containing n_segments + 1 vertices.
Vertex::Vector createRandomVertices(int maximum_derivative,
                                    size_t n_segments,
                                    const Eigen::VectorXd& minimum_position,
                                    const Eigen::VectorXd& maximum_position,
                                    size_t seed = 0);

Vertex::Vector createSquareVertices(int maximum_derivative,
                                    const Eigen::Vector3d& center,
                                    double side_length,
                                    int rounds);

// Conveninence function to create 1D vertices.
Vertex::Vector createRandomVertices1D(int maximum_derivative,
                                      size_t n_segments,
                                      double minimum_position,
                                      double maximum_position,
                                      size_t seed = 0);

}  // namespace poly_opt