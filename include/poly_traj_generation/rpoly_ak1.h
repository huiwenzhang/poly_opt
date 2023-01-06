#pragma once
#include <iostream>
#include <fstream>
#include <cctype>
#include <cmath>
#include <cfloat>

#include <Eigen/Eigen>

namespace poly_opt {
int findLastNonZeroCoeff(const Eigen::VectorXd& coeffs);

// vectorxcd means complex double type
bool findRootsJenkinsTraub(const Eigen::VectorXd& coeffs_increase, Eigen::VectorXcd* roots);
}  // namespace poly_opt