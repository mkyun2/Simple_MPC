#include "mpc.hpp"

#include <iostream>

Predictor::Predictor(int state_num, int ctl_num, int pred_horizon,
                     int ctl_horizon)
    : state_num_(state_num),
      ctl_num_(ctl_num),
      pred_horizon_(pred_horizon),
      ctl_horizon_(ctl_horizon) {
  pred_state_ = Eigen::VectorXd(pred_horizon_ * state_num_, 1);
  pred_input_ = Eigen::VectorXd(ctl_horizon_ * ctl_num_, 1);
}
void Predictor::computePredictionMatrix(Eigen::MatrixXd transition_matrix,
                                        Eigen::MatrixXd input_matrix) {
  Eigen::MatrixXd A_pow_trans = Eigen::MatrixXd::Identity(
      transition_matrix.rows(), transition_matrix.cols());

  pred_trans_matrix_ =
      Eigen::MatrixXd::Zero(pred_horizon_ * state_num_, state_num_);
  pred_input_matrix_ = Eigen::MatrixXd::Zero(pred_horizon_ * state_num_,
                                             ctl_horizon_ * ctl_num_);
  // 2x3 3x3
  for (int i = 0; i < pred_horizon_; i++) {
    A_pow_trans *= transition_matrix;
    pred_trans_matrix_.block(state_num_ * i, 0, state_num_, state_num_) =
        A_pow_trans;

    for (int j = 0; j < ctl_horizon_; j++) {
      Eigen::MatrixXd A_pow_ctl = Eigen::MatrixXd::Identity(
          transition_matrix.rows(), transition_matrix.cols());
      if (j > i) break;
      for (int k = 0; k < i - j; k++) {
        A_pow_ctl *= transition_matrix;
      }
      pred_input_matrix_.block(state_num_ * i, ctl_num_ * j, state_num_,
                               ctl_num_) = A_pow_ctl * input_matrix;
    }
  }

  std::cout << "pred_trans_matrix" << std::endl;
  std::cout << pred_trans_matrix_ << std::endl;
  std::cout << "pred_input_matrix" << std::endl;
  std::cout << pred_input_matrix_ << std::endl;
}
Eigen::MatrixXd Predictor::getPredTransMatrix() { return pred_trans_matrix_; }
Eigen::MatrixXd Predictor::getPredInputMatrix() { return pred_input_matrix_; }
Optimizer::Optimizer() {}

void Optimizer::addIneqConstraint(int N, int n_u, Eigen::VectorXd u_min,
                                  Eigen::VectorXd u_max, float penalty) {
  Optimizer::InequalityConstraint ineq;
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(N * n_u, N * n_u);
  ineq.A_ineq.resize(2 * N * n_u, N * n_u);
  ineq.A_ineq << I, -I;

  Eigen::VectorXd u_min_vec(N * n_u), u_max_vec(N * n_u);
  for (int i = 0; i < N; ++i) {
    u_min_vec.segment(i * n_u, n_u) = u_min;
    u_max_vec.segment(i * n_u, n_u) = u_max;
  }

  ineq.b_ineq.resize(2 * N * n_u);
  ineq.b_ineq << u_max_vec, -u_min_vec;
  ineq.penalty = penalty;
  IneqConstraints_.push_back(ineq);
}

void Optimizer::computePredInput(Eigen::MatrixXd pred_state,
                                 Eigen::MatrixXd pred_state_matrix,
                                 Eigen::MatrixXd pred_state_weight,
                                 Eigen::MatrixXd pred_input_matrix,
                                 Eigen::MatrixXd pred_input_weight,
                                 Eigen::VectorXd &pred_input) {
  Eigen::MatrixXd Hesse =
      (pred_input_matrix.transpose() * pred_state_weight * pred_input_matrix +
       pred_input_weight);
  Eigen::MatrixXd g =
      pred_input_matrix.transpose() * pred_state_weight * pred_state;
  // std::cout<< "inverse" << std::endl;
  // td::cout << coeff_.inverse().rows() << ", " << coeff_.inverse().cols() <<
  // '\n'; Eigen::MatrixXd constraint_force = (-g-Hesse*coeff_.inverse()*b_);
  double convergence = 1.0;
  double eta = 0.1;
  bool result = Optimizer::cholesky_decomposition(Hesse);
  if (result) {
    Eigen::VectorXd pred_input_ = pred_input;
    Eigen::VectorXd delta_input = pred_input;

    std::cout << "Newton Method Start" << std::endl;
    while (convergence > 10e-6) {
      // delta_input = all_Hess.inverse() * all_Jacobi;
      Eigen::MatrixXd Jacobi_constraints =
          Eigen::MatrixXd::Zero(pred_input_.rows(), pred_input_.cols());
      Eigen::MatrixXd Hess_constraints =
          Eigen::MatrixXd::Zero(pred_input_.rows(), pred_input_.rows());
      for (auto &constraint : IneqConstraints_) {
        Eigen::MatrixXd s = constraint.A_ineq * pred_input_ - constraint.b_ineq;
        Eigen::VectorXd inv_s = s.cwiseInverse();
        Eigen::VectorXd inv_s2 = inv_s.cwiseProduct(inv_s);
        Jacobi_constraints +=
            constraint.penalty * constraint.A_ineq.transpose() * inv_s;
        Hess_constraints += constraint.penalty * constraint.A_ineq.transpose() *
                            inv_s2.asDiagonal() * constraint.A_ineq;
      }
      Eigen::MatrixXd all_Jacobi =
          (Hesse * pred_input_ + g) - Jacobi_constraints;
      Eigen::MatrixXd all_Hess = Hesse + Hess_constraints;
      delta_input = all_Hess.ldlt().solve(-all_Jacobi);

      convergence = delta_input.transpose() * Hesse * delta_input;
      pred_input_ = pred_input_ + eta * delta_input;
    }
    pred_input = pred_input_;
    std::cout << "Optimized :: Convergence Rate : " << convergence << std::endl;
    std::cout << "=====Solution=====" << std::endl;
    std::cout << pred_input << std::endl;
  } else {
    std::cout << "Can't Optimize :: Not Positive Definite" << std::endl;
  }
}
bool Optimizer::cholesky_decomposition(Eigen::MatrixXd matrix) {
  if (matrix.rows() != matrix.cols() || matrix != matrix.transpose())
    return false;

  Eigen::MatrixXd L = Eigen::MatrixXd::Zero(matrix.rows(), matrix.cols());

  for (int i = 0; i < matrix.rows(); i++) {
    for (int j = 0; j < i; j++) {
      L(i, j) = 1 / L(j, j) * (matrix(i, j));
      for (int k = 0; k < j; k++) {
        L(i, j) -= L(i, k) * L(j, k);
      }
    }

    L(i, i) += matrix(i, i);

    for (int j = 0; j < i; j++) {
      L(i, i) -= L(i, j) * L(i, j);
    }
    L(i, i) = sqrt(L(i, i));
    if (L(i, i) == 0) return false;
  }
  // std::cout << "INPUT"<<std::endl;
  // std::cout << matrix << std::endl;
  // std::cout << "Cholesky Decomposition"<<std::endl;
  // std::cout << L << std::endl;
  // Eigen::LLT<Eigen::MatrixXd> llt(matrix);
  // llt.compute(matrix);
  // Eigen::MatrixXd check = llt.matrixL();
  return true;
}