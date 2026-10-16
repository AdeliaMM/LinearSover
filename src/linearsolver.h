#pragma once

#ifndef EIGEN_USE_BLAS
#define EIGEN_USE_BLAS
#endif
#ifndef EIGEN_USE_LAPACKE
#define EIGEN_USE_LAPACKE
#endif

#include <Eigen/Dense>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace linalg {

// ---------------------------------------------------------------------
// Вспомогательные типы
// ---------------------------------------------------------------------

/** Исключение решателя (несовместимые размеры, вырожденная матрица и т.п.). */
class SolverException : public std::runtime_error {
public:
  explicit SolverException(const std::string& what)
      : std::runtime_error(what) {}
};

/** Причина остановки итерационного процесса. */
enum class StopReason {
  Converged,     // достигнута заданная точность
  MaxIterations, // исчерпан лимит итераций
  Diverged,      // норма невязки/шага начала расти
  Breakdown      // деление на ~0, потеря SPD и т.п.
};

/** Результат итерационного метода. */
struct IterativeResult {
  Eigen::VectorXd x;  // решение
  int iterations = 0; // число итераций
  double residual = 0.0; // ||b - A x||_2 (или относительная невязка)
  StopReason reason = StopReason::MaxIterations;

  bool converged() const noexcept { return reason == StopReason::Converged; }
};

/** Параметры итерационных методов. */
struct IterativeOptions {
  double tolerance = 1e-10; // критерий остановки
  int maxIter = 10000;      // максимум итераций
  bool verbose = false;     // печать прогресса
  double relaxation = 1.0; // omega для SOR (1.0 -> чистый Зейдель)
  std::optional<Eigen::VectorXd> x0; // начальное приближение (nullopt -> нули)
};

// ---------------------------------------------------------------------
// Основной класс
// ---------------------------------------------------------------------

/**
 * @brief Решатель СЛАУ Ax = b.
 *
 * Внутри хранит плотные A (n x n) и b (n). Копирует входные данные, чтобы
 * решатель можно было переиспользовать с разными параметрами и методами.
 *
 * Стратегия:
 *   - прямые методы (LU, QR, LLT/LDLT) делегируются LAPACK через Eigen
 *     (EIGEN_USE_LAPACKE) и BLAS (EIGEN_USE_BLAS для gemm/gemv);
 *   - итерационные методы реализуются вручную поверх Eigen-выражений,
 *     чтобы иметь полный контроль над критериями остановки и диагностикой.
 */
class LinearSolver {
public:
  using Matrix = Eigen::MatrixXd;
  using Vector = Eigen::VectorXd;

  // ---------------------------------------------------------------
  // Конструирование
  // ---------------------------------------------------------------

  /**
   * @param A  матрица коэффициентов (n x n)
   * @param b  вектор правых частей (n)
   * @throws SolverException при несовпадении размеров или пустой системе
   */
  LinearSolver(const Matrix& A, const Vector& b);
  LinearSolver(Matrix&& A, Vector&& b);

  LinearSolver(LinearSolver&& ) noexcept = default;
  LinearSolver& operator=(LinearSolver&& ) noexcept = default;
  LinearSolver(const LinearSolver& ) = delete;
  LinearSolver& operator=(const LinearSolver& ) = delete;

  // ---------------------------------------------------------------
  // Информация о системе
  // ---------------------------------------------------------------

  Eigen::Index size() const noexcept; // n
  const Matrix& A() const noexcept;
  const Vector& b() const noexcept;

  /** Диагональное преобладание (необходимо для Якоби/Зейделя). */
  bool IsDiagonallyDominant(double eps = 1e-12) const noexcept;

  /** Симметричность (важно для CG и наискорейшего спуска). */
  bool IsSymmetric(double eps = 1e-12) const noexcept;

  /** Положительная определённость (проверка через LLT). */
  bool IsPositiveDefinite() const noexcept;

  /** Число обусловленности (оценка через SVD). */
  double ConditionNumber() const;

  // ---------------------------------------------------------------
  // Прямые методы (LAPACK через Eigen)
  // ---------------------------------------------------------------

  /**
   * LU-разложение с частичным выбором ведущего элемента.
   */
  Vector SolveLU() const;

  /**
   * LU с полным выбором ведущего элемента (устойчивее, дороже).
   */
  Vector SolveLUFullPivot() const;

  /**
   * QR-разложение (Householder).
   * Полезно для переопределённых/плохо обусловленных систем.
   */
  Vector SolveQR() const;

  /**
   * Разложение Холецкого (только для SPD-матриц).
   */
  Vector SolveLLT() const;

  /**
   * LDL^T-разложение (для симметричных, но не обязательно SPD).
   */
  Vector SolveLDLT() const;

  /**
   * Возвращает сами множители LU (для отладки/анализа).
   * @param L  нижняя треугольная с единицами на диагонали
   * @param U  верхняя треугольная
   * @param piv  перестановка строк (индексы)
   */
  void ComputeLU(Matrix &LU, Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> &P) const;
  void ComputeLU(Matrix &LU, Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> &P, Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> &Q) const;

  // ---------------------------------------------------------------
  // Итерационные методы (ручная реализация поверх Eigen)
  // ---------------------------------------------------------------

  /** Метод простых итераций: x_{k+1} = x_k + tau (b - A x_k). */
  IterativeResult SolveSimpleIteration(const IterativeOptions& opts = {}) const;

  /** Метод Якоби. */
  IterativeResult SolveJacobi(const IterativeOptions& opts = {}) const;

  /** Метод Зейделя (relaxation == 1) или SOR (relaxation != 1). */
  IterativeResult SolveSeidel(const IterativeOptions& opts = {}) const;

  /** Метод наискорейшего спуска (A должна быть SPD). */
  IterativeResult SolveSteepestDescent(const IterativeOptions& opts = {}) const;

  /** Метод сопряжённых градиентов (A должна быть SPD). */
  IterativeResult SolveConjugateGradient(const IterativeOptions& opts = {}) const;

  // ---------------------------------------------------------------
  // Утилиты
  // ---------------------------------------------------------------

  /** Невязка r = b - A x. */
  Vector Residual(const Vector& x) const;

  /** Относительная невязка ||b - A x|| / ||b||. */
  double RelativeResidual(const Vector& x) const;

  /** Матрично-векторное произведение A * x. */
  Vector MatVec(const Vector& x) const;

private:
  Eigen::Index n_;
  Matrix A_;
  Vector b_;
};

} // namespace linalg
