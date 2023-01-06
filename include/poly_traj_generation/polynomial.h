#pragma once

#include <assert.h>

#include <Eigen/Eigen>
#include <Eigen/SVD>
#include <limits>
#include <utility>
#include <vector>

namespace poly_opt {
// Implementation of polynomials of order N-1. Order must be known at
// compile time.
// Polynomial coefficients are stored with increasing powers,
// i.e. c_0 + c_1*t ... c_{N-1} * t^{N-1}
// where N = number of coefficients of the polynomial.
class Polynomial {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef std::vector<Polynomial> Vector;

    static constexpr int kMaxN = 12;
    static constexpr int kMaxConvolutionSize = 2 * kMaxN - 2;
    static Eigen::MatrixXd base_coefficients_;

    Polynomial(int N) : N_(N), coefficients_(N) { coefficients_.setZero(); }

    Polynomial(int N, const Eigen::VectorXd& coeffs) : N_(N), coefficients_(coeffs) {
        assert(N_ = coeffs.size());
    }
    Polynomial(const Eigen::VectorXd& coeffs) : N_(coeffs.size()), coefficients_(coeffs) {}

    int N() const { return N_; }

    inline bool operator==(const Polynomial& rhs) const {
        return coefficients_ == rhs.coefficients_;
    }
    inline bool operator!=(const Polynomial& rhs) const { return !operator==(rhs); }
    inline Polynomial operator+(const Polynomial& rhs) const {
        return Polynomial(coefficients_ + rhs.coefficients_);
    }
    inline Polynomial& operator+=(const Polynomial& rhs) {
        this->coefficients_ += rhs.coefficients_;
        return *this;
    }

    inline Polynomial operator*(const Polynomial& rhs) const {
        return Polynomial(convolve(coefficients_, rhs.coefficients_));
    }

    inline Polynomial operator*(const double& rhs) const { return Polynomial(coefficients_ * rhs); }

    void setCoefficients(const Eigen::VectorXd& coeffs) {
        assert(N_ = coeffs.size());
        coefficients_ = coeffs;
    }

    Eigen::VectorXd getCoefficients(int derivative = 0) const {
        assert(derivative <= N_);
        if (derivative == 0) {
            return coefficients_;
        } else {
            Eigen::VectorXd result(N_);
            result.setZero();
            result.head(N_ - derivative) =
                    coefficients_.tail(N_ - derivative)
                            .cwiseProduct(base_coefficients_
                                                  .block(derivative, derivative, 1, N_ - derivative)
                                                  .transpose());
            return result;
        }
    }

    void evaluate(double t, Eigen::VectorXd* result) const {
        assert(result->size() <= N_);
        const int max_deg = result->size();

        const int tmp = N_ - 1;
        for (int i = 0; i < max_deg; i++) {
            Eigen::RowVectorXd row = base_coefficients_.block(i, 0, 1, N_);
            double acc = row[tmp] * coefficients_[tmp];
            for (int j = tmp - 1; j >= i; --j) {
                acc *= t;
                acc += row[j] * coefficients_[j];
            }
            (*result)[i] = acc;
        }
    }

    // Evaluates the specified derivative of the polynomial at time t and returns
    // the result (only one value).
    double evaluate(double t, int derivative) const {
        if (derivative >= N_) { return 0.0; }
        double result;
        const int tmp = N_ - 1;
        Eigen::RowVectorXd row = base_coefficients_.block(derivative, 0, 1, N_);
        result = row[tmp] * coefficients_[tmp];
        for (int j = tmp - 1; j >= derivative; --j) {
            result *= t;
            result += row[j] * coefficients_[j];
        }
        return result;
    }

    // Uses Jenkins-Traub to get all the roots of the polynomial at a certain
    // derivative.
    bool getRoots(int derivative, Eigen::VectorXcd* roots) const;

    // Finds all candidates for the minimum and maximum between t_start and t_end
    // by evaluating the roots of the polynomial's derivative.
    static bool selectMinMaxCandidatesFromRoots(
            double t_start,
            double t_end,
            const Eigen::VectorXcd& roots_derivative_of_derivative,
            std::vector<double>* candidates);

    // Finds all candidates for the minimum and maximum between t_start and t_end
    // by computing the roots of the derivative polynomial.
    bool computeMinMaxCandidates(double t_start,
                                 double t_end,
                                 int derivative,
                                 std::vector<double>* candidates) const;

    // Evaluates the minimum and maximum of a polynomial between time t_start and
    // t_end given the roots of the derivative.
    // Returns the minimum and maximum as pair<t, value>.
    bool selectMinMaxFromRoots(double t_start,
                               double t_end,
                               int derivative,
                               const Eigen::VectorXcd& roots_derivative_of_derivative,
                               std::pair<double, double>* minimum,
                               std::pair<double, double>* maximum) const;

    // Computes the minimum and maximum of a polynomial between time t_start and
    // t_end by computing the roots of the derivative polynomial.
    // Returns the minimum and maximum as pair<t, value>.
    bool computeMinMax(double t_start,
                       double t_end,
                       int derivative,
                       std::pair<double, double>* minimum,
                       std::pair<double, double>* maximum) const;

    // Selects the minimum and maximum of a polynomial among a candidate set.
    // Returns the minimum and maximum as pair<t, value>.
    bool selectMinMaxFromCandidates(const std::vector<double>& candidates,
                                    int derivative,
                                    std::pair<double, double>* minimum,
                                    std::pair<double, double>* maximum) const;

    // Increase the number of coefficients of this polynomial up to the specified
    // degree by appending zeros.
    bool getPolynomialWithAppendedCoefficients(int new_N, Polynomial* new_polynomial) const;

    // Computes the base coefficients with the according powers of t, as
    // e.g. needed for computation of (in)equality constraints.
    // Output: coeffs = vector to write the coefficients to
    // Input: polynomial derivative for which the coefficients have to
    // be computed
    // Input: t = time of evaluation
    static void baseCoeffsWithTime(int N, int derivative, double t, Eigen::VectorXd* coeffs) {
        assert(derivative < N);
        assert(derivative >= 0);

        coeffs->resize(N, 1);
        coeffs->setZero();

        (*coeffs)[derivative] =
                base_coefficients_(derivative, derivative);  // base_coeffs第一个是常数项
        if (std::abs(t) < std::numeric_limits<double>::epsilon()) return;

        double t_power = t;
        for (int j = derivative + 1; j < N; j++) {
            (*coeffs)[j] = base_coefficients_(derivative, j) * t_power;
            t_power = t_power * t;
        }
    }

    static Eigen::VectorXd baseCoeffsWithTime(int N, int derivative, double t) {
        Eigen::VectorXd c(N);
        baseCoeffsWithTime(N, derivative, t, &c);
        return c;
    }

    static Eigen::VectorXd convolve(const Eigen::VectorXd& data, const Eigen::VectorXd& kernal);

    static inline int getConvolutionLength(int size, int kernal_size) {
        return size + kernal_size - 1;
    }

    // Scales the polynomial in time with a scaling factor.
    // To stretch out by a factor of 10, pass scaling_factor (b) = 1/10. To shrink
    // by a factor of 10, pass scalign factor = 10.
    // p_out = a12*b^12*t^12 + a11*b^11*t^11... etc.
    void scalePolynomialInTime(double scaling_factor);

    // Offset this polynomial.
    void offsetPolynomial(const double offset);

private:
    int N_;
    Eigen::VectorXd coefficients_;
};

// Computes the base coefficients of the derivatives of the polynomial,
// up to order N.
Eigen::MatrixXd computeBaseCoefficients(int N);

}  // namespace poly_opt