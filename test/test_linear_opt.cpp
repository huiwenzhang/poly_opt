#include <fstream>

#include "poly_traj_generation/io.h"
#include "poly_traj_generation/poly_opt_linear.h"
using namespace poly_opt;
using namespace Eigen;

int main(int argc, char** argv) {
   
    // 1. 生成waypoint和约束
    Vertex::Vector vertices;
    const int dimension = 3;
    const double v_max = 2.0;
    const double a_max = 2.0;
    const int derivative_to_opt = derivative_order::JERK;
    Vertex start(dimension), middle(dimension), end(dimension);

    start.makeStartOrEnd(Eigen::Vector3d(1, 0, 1), derivative_to_opt);
    vertices.push_back(start);

    middle.addConstraint(derivative_order::POSITION, Eigen::Vector3d(2, 0, 1));
    vertices.push_back(middle);

    end.makeStartOrEnd(Eigen::Vector3d(2, 1, 5), derivative_to_opt);
    vertices.push_back(end);

    // ************ random vertices *****************/
    vertices.clear();
    VectorXd pos_min(dimension), pos_max(dimension);
    pos_min.setConstant(-5);
    pos_max.setConstant(5);
    vertices = createRandomVertices(derivative_to_opt, 12, pos_min, pos_max, 4);

    if(saveVerticesToFile("./poly/vertices.txt", vertices)) {
        std::cout << "save vetex to succ\n";
    } else {
        std::cout << "failed to save vetex\n";
    }

    // 2. 计算每个segment用时
    std::vector<double> segment_times;
    segment_times = estimateSegmentTimes(vertices, v_max, a_max);
    // segment_times = estimateSegmentTimesVelocityRamp(vertices, v_max, a_max);

    // 3. 优化
    const int N = 8;  // for jerk
    PolyOptimization<N> opt(dimension);
    opt.setupFromVertices(vertices, segment_times, derivative_to_opt);
    opt.solveLinear();

    // Segment::Vector segments;
    // opt.getSegments(&segments);
    // for (auto& segment : segments) { std::cout << segment << std::endl; }

    // 4. 返回轨迹
    Trajectory traj;
    opt.getTrajectory(&traj);
    // minfo("total trajectory cost %f secs", traj.getMaxTime());
    saveTrajectoryToFile("./poly/linear_traj_", traj);

    return 0;
}