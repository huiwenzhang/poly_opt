#include "poly_traj_generation/vertex.h"

#include <assert.h>

#include <random>
namespace poly_opt {

Vertex::Vector createRandomVertices(int maximum_derivative,
                                    size_t n_segments,
                                    const Eigen::VectorXd& pos_min,
                                    const Eigen::VectorXd& pos_max,
                                    size_t seed) {
    assert(n_segments >= 1);
    assert(pos_min.size() == pos_max.size());
    assert((pos_max - pos_min).norm() >= 0.2);
    assert(maximum_derivative > 0);

    Vertex::Vector vertices;
    std::mt19937 generator(seed);
    std::vector<std::uniform_real_distribution<double> > distribution;

    const size_t dimension = pos_min.size();

    distribution.resize(dimension);

    for (size_t i = 0; i < dimension; ++i) {
        distribution[i] = std::uniform_real_distribution<double>(pos_min[i], pos_max[i]);
    }

    const double min_distance = 0.2;
    const size_t n_vertices = n_segments + 1;

    Eigen::VectorXd last_pos(dimension);
    for (size_t i = 0; i < dimension; ++i) { last_pos[i] = distribution[i](generator); }

    vertices.reserve(n_segments + 1);
    vertices.push_back(Vertex(dimension));

    vertices.front().makeStartOrEnd(last_pos, maximum_derivative);  // 第一个点

    for (size_t i = 1; i < n_vertices; ++i) {
        Eigen::VectorXd pos(dimension);

        while (true) {
            for (size_t d = 0; d < dimension; ++d) {
                pos[d] = distribution[d](generator);
            }
            if ((pos - last_pos).norm() > min_distance) { break; }
        }

        Vertex v(dimension);
        v.addConstraint(derivative_order::POSITION, pos);
        vertices.push_back(v);
        last_pos = pos;
    }
    vertices.back().makeStartOrEnd(last_pos, maximum_derivative);  // 最后一个点

    return vertices;
}

Vertex::Vector createSquareVertices(int maximum_derivative,
                                    const Eigen::Vector3d& center,
                                    double side_length,
                                    int rounds) {
    Vertex::Vector vertices;
    const size_t dimension = center.size();

    Eigen::Vector3d pos1(
            center[0] - side_length / 2.0, center[1] - side_length / 2.0, center[2]);
    Vertex v1(dimension);
    v1.addConstraint(derivative_order::POSITION, pos1);
    Eigen::Vector3d pos2(
            center[0] - side_length / 2.0, center[1] + side_length / 2.0, center[2]);
    Vertex v2(dimension);
    v2.addConstraint(derivative_order::POSITION, pos2);
    Eigen::Vector3d pos3(
            center[0] + side_length / 2.0, center[1] + side_length / 2.0, center[2]);
    Vertex v3(dimension);
    v3.addConstraint(derivative_order::POSITION, pos3);
    Eigen::Vector3d pos4(
            center[0] + side_length / 2.0, center[1] - side_length / 2.0, center[2]);
    Vertex v4(dimension);
    v4.addConstraint(derivative_order::POSITION, pos4);

    vertices.reserve(4 * rounds);
    vertices.push_back(v1);
    vertices.front().makeStartOrEnd(pos1, maximum_derivative);

    for (int i = 0; i < rounds; ++i) {
        vertices.push_back(v2);
        vertices.push_back(v3);
        vertices.push_back(v4);
        vertices.push_back(v1);
    }
    vertices.back().makeStartOrEnd(pos1, maximum_derivative);

    return vertices;
}

Vertex::Vector createRandomVertices1D(int maximum_derivative,
                                      size_t n_segments,
                                      double pos_min,
                                      double pos_max,
                                      size_t seed) {
    return createRandomVertices(maximum_derivative,
                                n_segments,
                                Eigen::VectorXd::Constant(1, pos_min),
                                Eigen::VectorXd::Constant(1, pos_max),
                                seed);
}

void Vertex::addConstraint(int derivative_order, const Eigen::VectorXd& constraint) {
    assert(constraint.rows() == static_cast<long>(D_));
    constraints_[derivative_order] = constraint;
}

bool Vertex::removeConstraint(int type) {
    Constraints::const_iterator it = constraints_.find(type);
    if (it != constraints_.end()) {
        constraints_.erase(it);
        return true;
    } else {
        // Constraint not found.
        return false;
    }
}

void Vertex::makeStartOrEnd(const Eigen::VectorXd& constraint, int up_to_derivative) {
    addConstraint(derivative_order::POSITION, constraint);
    for (int i = 1; i <= up_to_derivative; ++i) {
        constraints_[i] = ConstraintValue::Zero(static_cast<int>(D_));
    }
}

void Vertex::makeStartOrEnd(int up_to_derivative) {
    // addConstraint(derivative_order::POSITION, constraint);
    for (int i = 1; i <= up_to_derivative; ++i) {
        constraints_[i] = ConstraintValue::Zero(static_cast<int>(D_));
    }
}

bool Vertex::getConstraint(int derivative_order, Eigen::VectorXd* value) const {
    assert(value != nullptr);
    typename Constraints::const_iterator it = constraints_.find(derivative_order);
    if (it != constraints_.end()) {
        *value = it->second;
        return true;
    } else
        return false;
}

bool Vertex::hasConstraint(int derivative_order) const {
    typename Constraints::const_iterator it = constraints_.find(derivative_order);
    return it != constraints_.end();
}

bool Vertex::isEqualTol(const Vertex& rhs, double tol) const {
    if (constraints_.size() != rhs.constraints_.size()) return false;
    // loop through lhs constraint map
    for (typename Constraints::const_iterator it = cBegin(); it != cEnd(); ++it) {
        // look for matching key
        typename Constraints::const_iterator rhs_it = rhs.constraints_.find(it->first);
        if (rhs_it == rhs.constraints_.end()) return false;
        // check value
        if (!((it->second - rhs_it->second).isZero(tol))) return false;
    }
    return true;
}

bool Vertex::getSubdimension(const std::vector<size_t>& subdimensions,
                             int max_derivative_order,
                             Vertex* subvertex) const {
    assert(subvertex != nullptr);
    *subvertex = Vertex(subdimensions.size());

    // Check if all subdimensions exist.
    for (size_t subdimension : subdimensions)
        if (subdimension >= D_) return false;

    // Copy constraints up to maximum derivative order.
    for (Constraints::const_iterator it = constraints_.begin(); it != constraints_.end();
         ++it) {
        int derivative_order = it->first;
        if (derivative_order > max_derivative_order) continue;
        const ConstraintValue& original_constraint = it->second;
        ConstraintValue subsconstraint(subvertex->D());
        for (size_t i = 0; i < subdimensions.size(); i++) {
            subsconstraint[i] = original_constraint[subdimensions[i]];
        }
        subvertex->addConstraint(derivative_order, subsconstraint);
    }
    return true;
}

std::ostream& operator<<(std::ostream& stream, const Vertex& v) {
    stream << "constraints: " << std::endl;
    Eigen::IOFormat format(4, 0, ", ", "\n", "[", "]");
    for (typename Vertex::Constraints::const_iterator it = v.cBegin(); it != v.cEnd();
         ++it) {
        stream << "  type: " << positionDerivativeToString(it->first);
        stream << "  value: " << it->second.transpose().format(format) << std::endl;
    }
    return stream;
}

std::ostream& operator<<(std::ostream& stream, const std::vector<Vertex>& vertices) {
    for (const Vertex& v : vertices) { stream << v << std::endl; }
    return stream;
}

std::vector<double> estimateSegmentTimes(const Vertex::Vector& vertices,
                                         double v_max,
                                         double a_max) {
    return estimateSegmentTimesNfabian(vertices, v_max, a_max);
}

std::vector<double> estimateSegmentTimesVelocityRamp(const Vertex::Vector& vertices,
                                                     double v_max,
                                                     double a_max,
                                                     double time_factor) {
    assert(vertices.size() >= 2);
    std::vector<double> segment_times;

    segment_times.reserve(vertices.size() - 1);

    constexpr double kMinSegmentTime = 0.06;

    for (size_t i = 0; i < vertices.size() - 1; ++i) {
        Eigen::VectorXd start, end;
        vertices[i].getConstraint(derivative_order::POSITION, &start);
        vertices[i + 1].getConstraint(derivative_order::POSITION, &end);
        double t = computeTimeVelocityRamp(start, end, v_max, a_max);
        t = std::max(kMinSegmentTime, t);
        segment_times.push_back(t);
    }

    return segment_times;
}

std::vector<double> estimateSegmentOrientationTimesRamp(
        const std::vector<Eigen::Quaterniond>& vertices,
        double v_max,
        double a_max,
        double time_factor) {
    assert(vertices.size() >= 2);
    std::vector<double> segment_times;

    segment_times.reserve(vertices.size() - 1);

    constexpr double kMinSegmentTime = 0.06;

    for (size_t i = 0; i < vertices.size() - 1; ++i) {
        Eigen::Quaterniond start, end;
        start = vertices[i];
        end = vertices[i + 1];
        double t = computeTimeVelocityRamp(start, end, v_max, a_max);
        t = std::max(kMinSegmentTime, t);
        segment_times.push_back(t);
    }

    return segment_times;
}

std::vector<double> estimateSegmentTimesNfabian(const Vertex::Vector& vertices,
                                                double v_max,
                                                double a_max,
                                                double magic_fabian_constant) {
    assert(vertices.size() >= 2);
    std::vector<double> segment_times;
    segment_times.reserve(vertices.size() - 1);
    for (size_t i = 0; i < vertices.size() - 1; ++i) {
        Eigen::VectorXd start, end;
        vertices[i].getConstraint(derivative_order::POSITION, &start);
        vertices[i + 1].getConstraint(derivative_order::POSITION, &end);
        double distance = (end - start).norm();
        double t = distance / v_max * 2 *
                   (1.0 +
                    magic_fabian_constant * v_max / a_max * exp(-distance / v_max * 2));
        segment_times.push_back(t);
    }
    return segment_times;
}

double computeTimeVelocityRamp(const Eigen::VectorXd& start,
                               const Eigen::VectorXd& goal,
                               double v_max,
                               double a_max) {
    const double distance = (start - goal).norm();
    // Time to accelerate or decelerate to or from maximum velocity:
    const double acc_time = v_max / a_max;
    // Distance covered during complete acceleration or decelerate:
    const double acc_distance = 0.5 * v_max * acc_time;
    // Compute total segment time:
    if (distance < 2.0 * acc_distance) {
        // Case 1: Distance too small to accelerate to maximum velocity.
        return 2.0 * std::sqrt(distance / a_max);
    } else {
        // Case 2: Distance long enough to accelerate to maximum velocity.
        return 2.0 * acc_time + (distance - 2.0 * acc_distance) / v_max;
    }
}

double computeTimeVelocityRamp(const Eigen::Quaterniond& start,
                               const Eigen::Quaterniond& goal,
                               double av_max,
                               double aa_max) {
    const double distance = start.angularDistance(goal);  // unit: rad
    // Time to accelerate or decelerate to or from maximum velocity:
    const double acc_time = av_max / aa_max;
    // Distance covered during complete acceleration or decelerate:
    const double acc_distance = 0.5 * av_max * acc_time;
    // Compute total segment time:
    if (distance < 2.0 * acc_distance) {
        // Case 1: Distance too small to accelerate to maximum velocity.
        return 2.0 * std::sqrt(distance / aa_max);
    } else {
        // Case 2: Distance long enough to accelerate to maximum velocity.
        return 2.0 * acc_time + (distance - 2.0 * acc_distance) / av_max;
    }
}
}  // namespace poly_opt