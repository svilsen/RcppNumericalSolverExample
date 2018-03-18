#include <Rcpp.h>

//[[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

//[[Rcpp::depends(RcppNumericalSolver)]]
#include <cppoptlib/meta.h>
#include <cppoptlib/problem.h>
#include <cppoptlib/boundedproblem.h>
#include <cppoptlib/solver/bfgssolver.h>
#include <cppoptlib/solver/lbfgsbsolver.h>

class LinearRegression : public cppoptlib::Problem<double> {
public:
  using typename cppoptlib::Problem<double>::TVector;
  using MatrixType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

protected:
  const MatrixType X;
  const TVector y;
  const MatrixType XX;

public:
  LinearRegression(const MatrixType &X_, const TVector &y_) : X(X_), y(y_), XX(X_.transpose()*X_) {};
  ~LinearRegression() throw() {};

  double value(const TVector &beta)
  {
    return 0.5*(X*beta-y).squaredNorm();
  }

  void gradient(const TVector &beta, TVector &grad)
  {
    grad = XX*beta - X.transpose()*y;
  }
};

//' @title RcppNumericalSolver example
//'
//' @description An example demonstrating RcppNumericalSolver
//'
//' @return Numeric vector
//[[Rcpp::export()]]
Eigen::VectorXd linear_regression_example()
{
  typedef LinearRegression::TVector TVector;
  typedef LinearRegression::MatrixType MatrixType;

  // create true model
  TVector true_beta = TVector::Random(4);

  // create data
  MatrixType X = MatrixType::Random(50, 4);
  TVector y = X*true_beta;

  // perform linear regression
  LinearRegression f(X, y);

  TVector beta = TVector::Random(4);
  cppoptlib::BfgsSolver<LinearRegression> solver;
  solver.minimize(f, beta);

  return beta;
}

class Rosenbrock : public cppoptlib::Problem<double> {
public:
  using typename cppoptlib::Problem<double>::TVector;
  using typename cppoptlib::Problem<double>::THessian;

  double value(const TVector &x)
  {
    const double t1 = (1 - x[0]);
    const double t2 = (x[1] - x[0] * x[0]);
    return   t1 * t1 + 100 * t2 * t2;
  }

  void gradient(const TVector &x, TVector &grad) {
    grad[0] = -2 * (1 - x[0]) + 200 * (x[1] - x[0] * x[0]) * (-2 * x[0]);
    grad[1] = 200 * (x[1] - x[0] * x[0]);
  }

  void hessian(const TVector &x, THessian &hessian) {
    hessian(0, 0) = 1200 * x[0] * x[0] - 400 * x[1] + 1;
    hessian(0, 1) = -400 * x[0];
    hessian(1, 0) = -400 * x[0];
    hessian(1, 1) = 200;
  }

};

//[[Rcpp::export()]]
Eigen::VectorXd rosenbrock_example()
{
  Rosenbrock f;
  Eigen::VectorXd x(2); x << -1, 2;

  cppoptlib::BfgsSolver<Rosenbrock> solver;
  solver.minimize(f, x);

  return x;
}


class NonNegativeLeastSquares : public cppoptlib::BoundedProblem<double> {
  public :
    using Superclass = BoundedProblem<double>;
    using typename Superclass::TVector;
    using TMatrix = typename Superclass::THessian;

    const TMatrix X;
    const TVector y;

    NonNegativeLeastSquares(const TMatrix &X_, const TVector y_) :
      Superclass(X_.rows()),
      X(X_), y(y_) {}

    double value(const TVector &beta)
    {
      return (X*beta-y).dot(X*beta-y);
    }

    void gradient(const TVector &beta, TVector &grad)
    {
      grad = X.transpose()*2*(X*beta-y);
    }
};

//[[Rcpp::export()]]
Eigen::VectorXd nonnegativeleastsquares_example()
{
  const size_t DIM = 4;
  const size_t NUM = 10;
  typedef NonNegativeLeastSquares TNNLS;
  typedef typename TNNLS::TVector TVector;
  typedef typename TNNLS::TMatrix TMatrix;

  TMatrix X = TMatrix::Random(NUM, DIM);
  TVector true_beta = TVector::Random(DIM);
  TMatrix y = X*true_beta;

  TNNLS f(X, y);
  f.setLowerBound(TVector::Zero(DIM));

  TVector beta = TVector::Random(DIM);
  beta = (beta.array() < 0).select(-beta, beta);

  cppoptlib::LbfgsbSolver<TNNLS> solver;
  solver.minimize(f, beta);

  return beta;
}

