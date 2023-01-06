#include <fstream>

#include "poly_traj_generation/io.h"
#include "poly_traj_generation/poly_opt_linear.h"
using namespace poly_opt;
using namespace Eigen;

int main(int argc, char** argv) {
    // 1. 生成waypoint和约束
    Vertex::Vector vertices;
    const double v_max = 2.0;
    const double a_max = 2.0;
    const int derivative_to_opt = derivative_order::JERK;

    // ************ load vertices *****************/
    if (!verticesFromFile("car_waypoints.txt", derivative_to_opt, vertices)) { return 1; }
    const int dimension = vertices.front().D();

    saveVerticesToFile("poly/vertices.txt", vertices);

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

    saveTrajectoryToFile("./poly/results.txt", traj);

    return 0;
}