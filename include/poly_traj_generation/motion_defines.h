#pragma once

#include <string>

namespace poly_opt {
namespace derivative_order {
static constexpr int POSITION = 0;
static constexpr int VELOCITY = 1;
static constexpr int ACCELERATION = 2;
static constexpr int JERK = 3;
static constexpr int SNAP = 4;

static constexpr int ORIENTATION = 0;
static constexpr int ANGULAR_VELOCITY = 1;
static constexpr int ANGULAR_ACCELERATION = 2;

static constexpr int INVALID = -1;
}  // namespace derivative_order

std::string positionDerivativeToString(int derivative);
int positionDerivativeToInt(const std::string& string);

std::string orintationDerivativeToString(int derivative);
int orientationDerivativeToInt(const std::string& string);
}  // namespace poly_opt