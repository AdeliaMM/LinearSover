#include "linearsolver.h"

namespace linalg {

namespace {
constexpr double kEps = std::numeric_limits<double>::epsilon();
}

LinearSolver::LinearSolver(const Matrix& A, const Vector& b) : n_(A.rows()), A_(A), b_(b) {
  if ((n_ != A.columns()) || (n_ != b.rows())) {
    throw SolverException{};
  }
}

LinearSolver::LinearSolver(Matrix&& A, Vector&& b) : n_(A.rows()), A_(std::move(A)), b_(std::move(b)) {
  if ((n_ != A.columns()) || (n_ != b.rows())) {
    throw SolverException{};
  }
}

Eigen::Index LinearSolver::size() const noexcept {
  return size_;
}

const LinearSolver::Matrix& LinearSolver::A() const noexcept {
  return A_;
}

const LinearSolver::Vector& LinearSolver::b() const noexcept {
  return b_;
}

bool LinearSolver::IsDiagonallyDominant(double eps) const noexcept {
}

bool LinearSolver::IsSymmetric(double eps) const noexcept {
}

bool LinearSolver::IsPositiveDefinite() const noexcept {
}

double LinearSolver::ConditionNumber() const {
}

LinearSolver::Vector LinearSolver::SolveLU() const {
  Matrix LU;
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P;
  ComputeLU(LU, P);
  Vector v = Vector::Zero(n_);
  Vector pb = P * b_;
  for (Eigen::Index i = 0; i < n_; ++i) {
    v[i] = pb[i] - LU.triangularView<Eigen::UnitLower>().row(i).dot(v);
  }
  Vector u = Vector::Zero(n_);
  for (Eigen::Index i = n_ - 1; i-- > 0;) {
    u[i] = (v[i] - LU.triangularView<Eigen::Upper>().row(i).dot(u)) / LU(i, i);
  }
  return u;
}

LinearSolver::Vector LinearSolver::SolveLUFullPivot() const {
  Matrix LU;
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P;
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> Q;
  ComputeLU(LU, P, Q);
  Vector v = Vector::Zero(n_);
  Vector pb = P * b_;
  for (Eigen::Index i = 0; i < n_; ++i) {
    v[i] = pb[i] - LU.triangularView<Eigen::UnitLower>().row(i).dot(v);
  }
  Vector u = Vector::Zero(n_);
  for (Eigen::Index i = n_ - 1; i-- > 0;) {
    u[i] = (v[i] - LU.triangularView<Eigen::Upper>().row(i).dot(u)) / LU(i, i);
  }
  return Q * u;
}

void LinearSolver::ComputeLU(Matrix &LU, Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> &P) const {
  Eigen::PartialPivLU<Matrix> LU_(A_);
  LU = LU_.matrixLU();
  P = LU_.permutationP();
}

void LinearSolver::ComputeLU(Matrix &LU, Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> &P, Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> &Q) const {
  Eigen::FullPivLU<Matrix> LU_(A_);
  LU = LU_.matrixLU();
  P = LU_.permutationP();
  Q = LU_.permutationQ();
}

LinearSolver::Vector LinearSolver::SolveQR() const {
}

LinearSolver::Vector LinearSolver::SolveLLT() const {
}

LinearSolver::Vector LinearSolver::SolveLDLT() const {
}

LinearSolver::IterativeResult
LinearSolver::SolveSimpleIteration(const IterativeOptions& opts) const {
}

LinearSolver::IterativeResult
LinearSolver::SolveJacobi(const IterativeOptions& opts) const {
}

LinearSolver::IterativeResult
LinearSolver::SolveSeidel(const IterativeOptions& opts) const {
}

LinearSolver::IterativeResult
LinearSolver::SolveSteepestDescent(const IterativeOptions& opts) const {
}

LinearSolver::IterativeResult
LinearSolver::SolveConjugateGradient(const IterativeOptions& opts) const {
}

LinearSolver::Vector LinearSolver::Residual(const Vector& x) const {
}

double LinearSolver::RelativeResidual(const Vector& x) const {
}

LinearSolver::Vector LinearSolver::MatVec(const Vector& x) const {
}

} // namespace linalg
