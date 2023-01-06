#pragma once

#include <Eigen/Core>
#include <vector>

#include "poly_traj_generation/trajectory.h"
#include "poly_traj_generation/vertex.h"

namespace poly_opt {
bool saveVerticesToFile(const std::string& file, const Vertex::Vector& vertices);
bool saveTrajectoryToFile(const std::string& filename, const Trajectory& traj, int derivative = 0);
bool saveTrajectoryPVA(const std::string& filename, const Trajectory& traj);
bool verticesFromFile(const std::string& filename, int derivative_to_opt, Vertex::Vector& vertices);
}  // namespace poly_opt