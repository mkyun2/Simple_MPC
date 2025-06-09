#include <iostream>

#include "mpc.hpp"

int main() {
  int state_num = 2;
  int ctl_num = 1;
  int pred_horizon = 10;
  int ctl_horizon = 5;
  // def Reference and Init State
  Eigen::VectorXd state = Eigen::VectorXd(state_num);
  Eigen::VectorXd ref = Eigen::VectorXd(state_num);

  state << 0, 0;
  ref << 3, 0;
  // def Reference to predicted state
  Eigen::VectorXd x_ref = Eigen::VectorXd::Zero(pred_horizon * state_num);
  for (int i = 0; i < pred_horizon; ++i)
    x_ref.segment(i * state_num, state_num) = ref;
  // def Predictor for state prediction
  Predictor predictor(state_num, ctl_num, pred_horizon, ctl_horizon);
  Eigen::MatrixXd trans_matrix = Eigen::MatrixXd(state_num, state_num);
  Eigen::MatrixXd input_matrix = Eigen::MatrixXd(state_num, ctl_num);

  trans_matrix << 1, 1, 0, 1;
  input_matrix << 0.5, 1;
  double state_weight = 10.0;
  double input_weight = 1.0;
  // compute predicted Matrix
  predictor.computePredictionMatrix(trans_matrix, input_matrix);

  Eigen::MatrixXd pred_trans_matrix = predictor.getPredTransMatrix();
  Eigen::MatrixXd pred_input_matrix = predictor.getPredInputMatrix();
  Eigen::MatrixXd pred_state_weight = Eigen::MatrixXd::Identity(
      pred_horizon * state_num, pred_horizon * state_num) * state_weight;
  Eigen::MatrixXd pred_input_weight =
      Eigen::MatrixXd::Identity(ctl_horizon * ctl_num, ctl_horizon * ctl_num) *
      input_weight;
  Eigen::VectorXd pred_input = Eigen::VectorXd::Zero(ctl_horizon * ctl_num);

  /*
      Eigen::MatrixXd state_weight =
     Eigen::MatrixXd::Identity(state_num,state_num); Eigen::MatrixXd
     input_weight = Eigen::MatrixXd::Identity(ctl_num,ctl_num);
      pred_trans_matrix_ = Eigen::MatrixXd::Zero(pred_horizon_ * state_num_,
     state_num_); pred_input_matrix_ = Eigen::MatrixXd::Zero(pred_horizon_ *
     state_num_, ctl_horizon_ * ctl_num_);

      Eigen::MatrixXd pred_state = pred_trans_matrix * state + pred_input_matrix
     * pred_input; pred_horizion * state_num x state_num  * state num x 1 +
     pred_horizon * state_num x ctl_horizon * ctl_num  * ctl_horizon * ctl_ num
     x 1 ,
      => pred_state(pred_horizion * state_num, 1)
  */
  // def Optimizer to get optimal control
  Optimizer opt;
  /*inequality constraint g(x) = Ax - b -> coeff * x - b  => coeff_bar * x -
   * b_bar , condense many g(x) to consider prediction matrix */
  // set Inequality Constraints
  int constraint_num = 2;
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(ctl_num, ctl_num);
  Eigen::MatrixXd Aineq =
      Eigen::MatrixXd::Zero(ctl_num * constraint_num, ctl_num);
  Eigen::VectorXd b_max = Eigen::VectorXd::Zero(ctl_num);
  Eigen::VectorXd b_min = Eigen::VectorXd::Zero(ctl_num);
  b_max << 0.4;
  b_min << -0.4;

  opt.addIneqConstraint(ctl_horizon, ctl_num, b_min, b_max,2.0);

  for (int i = 0; i < 10; i++) {
    Eigen::VectorXd error = pred_trans_matrix * state - x_ref;

    if(!opt.computePredInput(error, pred_trans_matrix, pred_state_weight,
                         pred_input_matrix, pred_input_weight, pred_input))
        pred_input = 0.0 * pred_input;
    state = trans_matrix * state + input_matrix * pred_input(0);
    std::cout << "=====State=====" << std::endl;
    std::cout << state << std::endl;
  }

  return 0;
}