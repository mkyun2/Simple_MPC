#include "model.hpp"
#include "utils.hpp"


Model::Model(int state_num, int ctl_num)
    : state_num_(state_num), ctl_num_(ctl_num) {
  transition_matrix_.resize(state_num, state_num);
  state_.resize(state_num, 1);
  input_matrix_.resize(state_num, ctl_num);
  input_.resize(ctl_num, 1);
}
Eigen::MatrixXd Model::getTransMatrix() { return transition_matrix_; }
Eigen::MatrixXd Model::getInputMatrix() { return input_matrix_; }
Eigen::VectorXd Model::getState() { return state_; }
void Model::init(Eigen::VectorXd init_state)
{
  state_ = init_state;
}
void Model::update(Eigen::VectorXd input, double dt) {
  Eigen::VectorXd next_state;
  // clang-format off
  transition_matrix_ << 1, 0, 0, cos(state_(2))*dt, 0,
                        0, 1, 0, sin(state_(2))*dt, 0,
                        0, 0, 1, 0, dt,
                        0, 0, 0, 1, 0,
                        0, 0, 0, 0, dt;
  input_matrix_ << 0, 0,
                   0, 0,
                   0, 0,
                   dt, 0,
                   0, dt;
  // clang-format on
  
  next_state = transition_matrix_ * state_ + input_matrix_ * input;
  state_ = next_state;

}

void Model::setModel(Eigen::MatrixXd transition_matrix,
                     Eigen::MatrixXd input_matrix) {
  transition_matrix_ = transition_matrix;
  input_matrix_ = input_matrix;
}
Model::model Model::getLinearizedModel(const Eigen::VectorXd &ref) {
  Eigen::MatrixXd transition_matrix;
  Eigen::MatrixXd input_matrix;
  // clang-format off
  transition_matrix.resize(state_num_, state_num_);
  transition_matrix << 0, 0, -ref(3) * sin(ref(2)), cos(ref(2)), 0,
                       0, 0, ref(3)*cos(ref(2)), sin(ref(2)), 0,
                       0, 0, 0, 0, 1,
                       0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0;
  input_matrix.resize(state_num_, ctl_num_);
  input_matrix << 0, 0,
                  0, 0,
                  0, 0,
                  1, 0,
                  0, 1;
  // clang-format on

  Model::model linear_Model = {transition_matrix, input_matrix};
  return linear_Model;
}
Model::model Model::getDiscretizedModel(const model &model_info, int n,
                                        double dt) {
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(
      model_info.transition_matrix.rows(), model_info.transition_matrix.cols());

  Eigen::MatrixXd exp = I;
  for (int i = 1; i <= n; i++) {
    exp += powMatrix(model_info.transition_matrix, i) / factorial(i);
  }

  Eigen::MatrixXd discrete_transition_matrix = exp;

  Eigen::MatrixXd discrete_input_matrix = Eigen::MatrixXd(
      model_info.input_matrix.rows(), model_info.input_matrix.cols());
  discrete_input_matrix *= 0;

  for (int i = 1; i <= n; i++) {
    discrete_input_matrix += dt * i / factorial(i) *
                             powMatrix(model_info.transition_matrix, i - 1) *
                             model_info.input_matrix;
  }

  model discrete_model = {discrete_transition_matrix, discrete_input_matrix};
  return discrete_model;
}