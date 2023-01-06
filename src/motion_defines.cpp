#include "poly_traj_generation/motion_defines.h"

namespace poly_opt {

std::string positionDerivativeToString(int derivative) {
    if (derivative >= derivative_order::POSITION && derivative <= derivative_order::SNAP) {
        static constexpr const char* text[] = {
                "position", "velocity", "acceleration", "jerk", "snap"};
        return std::string(text[derivative]);
    } else {
        return std::string("invalid");
    }
}

int positionDerivativeToInt(const std::string& string) {
    using namespace derivative_order;
    if (string == "position") {
        return POSITION;
    } else if (string == "velocity") {
        return VELOCITY;
    } else if (string == "acceleration") {
        return ACCELERATION;
    } else if (string == "jerk") {
        return JERK;
    } else if (string == "snap") {
        return SNAP;
    } else {
        return INVALID;
    }
}

std::string orintationDerivativeToString(int derivative) {
    if (derivative >= derivative_order::ORIENTATION &&
        derivative <= derivative_order::ANGULAR_ACCELERATION) {
        static constexpr const char* text[] = {
                "orientation", "angular_velocity", "angular_acceleration"};
        return std::string(text[derivative]);
    } else {
        return std::string("invalid");
    }
}

int orientationDerivativeToInt(const std::string& string) {
    using namespace derivative_order;
    if (string == "orientation") {
        return ORIENTATION;
    } else if (string == "angular_velocity") {
        return ANGULAR_VELOCITY;
    } else if (string == "angular_acceleration") {
        return ANGULAR_ACCELERATION;
    } else {
        return INVALID;
    }
}

}  // namespace poly_opt