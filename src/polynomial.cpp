#include "poly_traj_generation/polynomial.h"

#include <algorithm>

// #include "log_manager.h"
#include "poly_traj_generation/rpoly_ak1.h"

namespace poly_opt {
Eigen::MatrixXd computeBaseCoefficients(int N) {
    Eigen::MatrixXd base_coefficients(N, N);

    base_coefficients.setZero();
    base_coefficients.row(0).setOnes();

    const int DEG = N - 1;  // order of polynominal
    int order = DEG;
    for (int n = 1; n < N; n++) {
        for (int i = DEG - order; i < N; i++) {
            base_coefficients(n, i) = (order - DEG + i) * base_coefficients(n - 1, i);
        }
        order--;
    }
    return base_coefficients;
}

bool Polynomial::getRoots(int derivative, Eigen::VectorXcd* roots) const {
    return findRootsJenkinsTraub(getCoefficients(derivative), roots);
}

bool Polynomial::selectMinMaxCandidatesFromRoots(
        double t_start,
        double t_end,
        const Eigen::VectorXcd& roots_derivative_of_derivative,
        std::vector<double>* candidates) {
    if (candidates == nullptr) return false;

    if (t_start > t_end) {
        // mwarn("t_start is greater than t_end");
        return false;
    }

    candidates->clear();
    candidates->reserve(roots_derivative_of_derivative.size() + 2);
    candidates->push_back(t_start);
    candidates->push_back(t_end);

    for (size_t i = 0; i < static_cast<size_t>(roots_derivative_of_derivative.size()); i++) {
        if (std::abs(roots_derivative_of_derivative[i].imag()) >
            std::numeric_limits<double>::epsilon()) {
            continue;
        }
        const double candidate = roots_derivative_of_derivative[i].real();
        if (candidate < t_start || candidate > t_end) {
            continue;
        } else {
            candidates->push_back(candidate);
        }
    }
    return true;
}

bool Polynomial::computeMinMaxCandidates(double t_start,
                                         double t_end,
                                         int derivative,
                                         std::vector<double>* candidates) const {
    if (candidates == nullptr) return false;
    candidates->clear();
    if (N_ - derivative - 1 < 0) {
        // mwarn("derivative exceeds the highest order");
        return false;
    }
    Eigen::VectorXcd roots;
    bool succ = getRoots(derivative + 1, &roots);
    if (!succ) {
        // minfo("Conldn't find roots, poly maybe constant");
        return false;
    }
    if (!selectMinMaxCandidatesFromRoots(t_start, t_end, roots, candidates)) {
        return false;
    } else {
        return true;
    }
}

bool Polynomial::selectMinMaxFromRoots(double t_start,
                                       double t_end,
                                       int derivative,
                                       const Eigen::VectorXcd& roots_derivative_of_derivative,
                                       std::pair<double, double>* minimum,
                                       std::pair<double, double>* maximum) const {
    if (minimum == nullptr || maximum == nullptr) return false;
    // Find candidates in interval t_start to t_end computing the roots.
    std::vector<double> candidates;
    if (!selectMinMaxCandidatesFromRoots(
                t_start, t_end, roots_derivative_of_derivative, &candidates)) {
        return false;
    }
    // Evaluate minimum and maximum.
    return selectMinMaxFromCandidates(candidates, derivative, minimum, maximum);
}

bool Polynomial::computeMinMax(double t_start,
                               double t_end,
                               int derivative,
                               std::pair<double, double>* minimum,
                               std::pair<double, double>* maximum) const {
    if (minimum == nullptr || maximum == nullptr) return false;

    // Find candidates in interval t_start to t_end by computing the roots.
    std::vector<double> candidates;
    if (!computeMinMaxCandidates(t_start, t_end, derivative, &candidates)) { return false; }
    // Evaluate minimum and maximum.
    return selectMinMaxFromCandidates(candidates, derivative, minimum, maximum);
}

bool Polynomial::selectMinMaxFromCandidates(const std::vector<double>& candidates,
                                            int derivative,
                                            std::pair<double, double>* minimum,
                                            std::pair<double, double>* maximum) const {
    if (minimum == nullptr || maximum == nullptr) return false;

    if (candidates.empty()) {
        // mwarn("Cannot find extrema from an empty candidates vector.");
        return false;
    }
    minimum->first = candidates[0];
    minimum->second = std::numeric_limits<double>::max();
    maximum->first = candidates[0];
    maximum->second = std::numeric_limits<double>::lowest();

    for (const double& t : candidates) {
        const double value = evaluate(t, derivative);
        if (value < minimum->second) {
            minimum->first = t;
            minimum->second = value;
        }
        if (value > maximum->second) {
            maximum->first = t;
            maximum->second = value;
        }
    }
    return true;
}

Eigen::VectorXd Polynomial::convolve(const Eigen::VectorXd& data, const Eigen::VectorXd& kernel) {
    const int convolution_dimension = getConvolutionLength(data.size(), kernel.size());
    Eigen::VectorXd convolved = Eigen::VectorXd::Zero(convolution_dimension);
    Eigen::VectorXd kernel_reverse = kernel.reverse();

    for (int i = 0; i < convolution_dimension; i++) {
        const int data_idx = i - kernel.size() + 1;

        int lower_bound = std::max(0, -data_idx);
        int upper_bound = std::min(kernel.size(), data.size() - data_idx);

        for (int kernel_idx = lower_bound; kernel_idx < upper_bound; ++kernel_idx) {
            convolved[i] += kernel_reverse[kernel_idx] * data[data_idx + kernel_idx];
        }
    }
    return convolved;
}

bool Polynomial::getPolynomialWithAppendedCoefficients(int new_N,
                                                       Polynomial* new_polynomial) const {
    if (new_N == N_) {
        *new_polynomial = *this;
        return true;
    } else if (new_N < N_) {
        // mwarn("You shan't decrease the number of coefficients.");
        *new_polynomial = *this;
        return false;
    } else {
        Eigen::VectorXd coeffs = Eigen::VectorXd::Zero(new_N);
        coeffs.head(N_) = coefficients_;
        *new_polynomial = Polynomial(coeffs);
        return true;
    }
}

void Polynomial::scalePolynomialInTime(double scaling_factor) {
    double scale = 1.0;
    for (int n = 0; n < N_; n++) {
        coefficients_[n] *= scale;
        scale *= scaling_factor;
    }
}

void Polynomial::offsetPolynomial(const double offset) {
    if (coefficients_.size() == 0) return;

    coefficients_[0] += offset;
}

Eigen::MatrixXd Polynomial::base_coefficients_ =
        computeBaseCoefficients(Polynomial::kMaxConvolutionSize);

}  // namespace poly_opt