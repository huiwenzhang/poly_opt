#include <fstream>

#include "poly_traj_generation/io.h"
#include "poly_traj_generation/poly_opt_nonlinear.h"
using namespace poly_opt;
using namespace Eigen;

int main(int argc, char** argv) {
    // 1. 生成waypoint和约束
    Vertex::Vector vertices;
    const int dimension = 3;
    const double v_max = 5.0;
    const double a_max = 5.0;
    const int derivative_to_opt = derivative_order::JERK;

    // ************ random vertices *****************/
    vertices.clear();
    VectorXd pos_min(dimension), pos_max(dimension);
    pos_min.setConstant(-5);
    pos_max.setConstant(5);
    vertices = createRandomVertices(derivative_to_opt, 3, pos_min, pos_max, 4);

    saveVerticesToFile("poly/nonlinear_vertices.txt", vertices);

    // 2. 计算每个segment用时
    std::vector<double> segment_times;
    segment_times = estimateSegmentTimes(vertices, v_max, a_max);
    // segment_times = estimateSegmentTimesVelocityRamp(vertices, v_max, a_max);

    // opt params config
    NonlinearOptimizationParameters params;
    params.max_iters = 3000;
    params.f_rel = 0.05;
    params.x_rel = 0.1;
    params.time_penalty = 500.0;
    params.initial_stepsize_rel = 0.1;
    params.inequality_constraint_tolerance = 0.1;
    params.algorithm = nlopt::LN_BOBYQA;
    params.random_seed = 124344;
    params.time_alloc_method = NonlinearOptimizationParameters::kSquaredTime;

    int ret;

    // 3. 优化，只优化时间
    const int N = 8;  // for jerk
    PolynomialOptimizationNonLinear<N> opt(dimension, params);
    opt.setupFromVertices(vertices, segment_times, derivative_to_opt);

    // 添加不等式约束
    opt.addMaximumMagnitudeConstraint(derivative_order::VELOCITY, v_max);
    opt.addMaximumMagnitudeConstraint(derivative_order::ACCELERATION, a_max);

    opt.solveLinear();
    double initial_cost = opt.getTotalCostWithSoftConstraints();
    //     minfo("initial cost %.3f", initial_cost);

    // 4. 返回轨迹
    Trajectory traj;
    double real_vmax, real_amax;
    opt.getTrajectory(&traj);

    traj.computeMaxVelocityAndAcceleration(&real_vmax, &real_amax);
    //     minfo("origin trajectory time %f secs, max vel: %.2f, max_acc: %.2f",
    //           traj.getMaxTime(),
    //           real_vmax,
    //           real_amax);
    saveTrajectoryPVA("poly/nfabian_linear.txt", traj);

    // minfo("==============sovle nonlinear=============");
    ret = opt.optimize();
    double final_cost = opt.getTotalCostWithSoftConstraints();

    //     minfo("nlopt1 stopped for reason: %s", nlopt::returnValueToString(ret).c_str());
    //     minfo("nlopt1 cost: %.3f", final_cost);

    // 4. 返回轨迹
    opt.getTrajectory(&traj);
    traj.computeMaxVelocityAndAcceleration(&real_vmax, &real_amax);
    //     minfo("opt time trajectory time %f secs, max vel: %.2f, max_acc: %.2f",
    //           traj.getMaxTime(),
    //           real_vmax,
    //           real_amax);
    saveTrajectoryPVA("poly/time_opt.txt", traj);

    /////////////////time & constraints //////////
    params.time_alloc_method = NonlinearOptimizationParameters::kSquaredTimeAndConstraints;
    params.print_debug_info = true;
    params.print_debug_info_time_allocation = true;
    PolynomialOptimizationNonLinear<N> opt1(dimension, params);
    opt1.setupFromVertices(vertices, segment_times, derivative_to_opt);

    // 添加不等式约束
    opt1.addMaximumMagnitudeConstraint(derivative_order::VELOCITY, v_max);
    opt1.addMaximumMagnitudeConstraint(derivative_order::ACCELERATION, a_max);

    ret = opt.optimize();
    double final_cost2 = opt.getTotalCostWithSoftConstraints();

    //     minfo("nlopt2 stopped for reason: %s", nlopt::returnValueToString(ret).c_str());
    //     minfo("nlopt2 cost: %.3f", final_cost2);
    // 4. 返回轨迹
    opt.getTrajectory(&traj);
    traj.computeMaxVelocityAndAcceleration(&real_vmax, &real_amax);
    //     minfo("opt time & free cons trajectory time %f secs, max vel: %.2f, max_acc: %.2f",
    //           traj.getMaxTime(),
    //           real_vmax,
    //           real_amax);
    saveTrajectoryPVA("poly/time_free_cons_opt.txt", traj);

    // test mellinger
    segment_times = estimateSegmentTimesVelocityRamp(vertices, v_max, a_max, 1.0);
    params.algorithm = nlopt::LD_LBFGS;
    params.time_alloc_method = NonlinearOptimizationParameters::kMellingerOuterLoop;
    params.print_debug_info = true;
    params.random_seed = 33455333;
    PolynomialOptimizationNonLinear<N> opt2(dimension, params);
    opt2.setupFromVertices(vertices, segment_times, derivative_to_opt);

    // 添加不等式约束
    opt2.addMaximumMagnitudeConstraint(derivative_order::VELOCITY, v_max);
    opt2.addMaximumMagnitudeConstraint(derivative_order::ACCELERATION, a_max);

    opt2.solveLinear();

    opt2.getTrajectory(&traj);
    traj.computeMaxVelocityAndAcceleration(&real_vmax, &real_amax);
    //     minfo("origin ramp time trajectory time %f secs, max vel: %.2f, max_acc: %.2f",
    //           traj.getMaxTime(),
    //           real_vmax,
    //           real_amax);
    saveTrajectoryPVA("poly/ramp_linear.txt", traj);

    opt2.scaleSegmentTimesWithViolation();

    opt2.getTrajectory(&traj);
    //     minfo("scale trajectory time %f secs, max vel: %.2f, max_acc: %.2f",
    //           traj.getMaxTime(),
    //           traj.getMaximumMagnitude(1),
    //           traj.getMaximumMagnitude(2));
    saveTrajectoryPVA("poly/ramp_scale.txt", traj);

    opt2.optimize();
    double mell_cost = opt2.getCost();
    opt2.getTrajectory(&traj);
    traj.computeMaxVelocityAndAcceleration(&real_vmax, &real_amax);
    //     minfo("mellinger trajectory time %f secs, max vel: %.2f, max_acc: %.2f",
    //           traj.getMaxTime(),
    //           real_vmax,
    //           real_amax);
    saveTrajectoryPVA("poly/mellinger_opt.txt", traj);

    return 0;
}