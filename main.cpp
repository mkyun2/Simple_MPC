#include <iostream>

#include "mpc.hpp"
#include "utils.hpp"
Eigen::VectorXd generate_reference_trajectory(int Np, double dt, int state_num,
                                              double current_time) {
  Eigen::VectorXd x_ref = Eigen::VectorXd::Zero(Np * state_num);

  // 사인 곡선 경로 파라미터 정의
  double amplitude = 2.0;  // 진폭
  double frequency = 0.5;  // 주파수
  double velocity = 1.0;   // 전진 속도

  for (int i = 0; i < Np; ++i) {
    Eigen::VectorXd ref_state = Eigen::VectorXd::Zero(state_num);
    double future_time = current_time + i * dt;

    // 1. 참조 위치 (x, y) 계산
    double ref_x = velocity * future_time;
    double ref_y = amplitude * sin(frequency * future_time);

    // 2. 참조 각도 (theta) 계산: 경로의 접선 각도
    // dx/dt = velocity
    // dy/dt = amplitude * frequency * cos(frequency * future_time)
    double ref_theta =
        atan2(amplitude * frequency * cos(frequency * future_time), velocity);

    // 3. 참조 상태 벡터 채우기 (x, y, theta, v, w)
    // 이 예제에서는 v, w는 0으로 가정하나, 필요시 목표 속도를 지정할 수
    // 있습니다.
    ref_state << ref_x, ref_y, ref_theta, 0, 0;

    x_ref.segment(i * state_num, state_num) = ref_state;
  }
  return x_ref;
}
int main() {
  int state_num = 5;
  int ctl_num = 2;
  int pred_horizon = 10;
  int ctl_horizon = 5;
  double dt = 0.05;
  double global_time = 0.0;
  // def Reference and Init State
  Model robot_model(state_num, ctl_num);

  std::vector<Eigen::VectorXd> path = genPath(2.0, 0.05);
  
  robot_model.init(Eigen::VectorXd::Zero(state_num));
  Eigen::VectorXd state = Eigen::VectorXd(state_num);
  Eigen::VectorXd ref = Eigen::VectorXd(state_num);
  state = robot_model.getState();

  // def Predictor for state prediction
  Predictor predictor(state_num, ctl_num, pred_horizon, ctl_horizon);
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
  b_max << 3.0, 3.0;
  b_min << -3.0, -3.0;

  opt.addIneqConstraint(ctl_horizon, ctl_num, b_min, b_max, 2.0);

  // double state_weight = 10.0;
  double input_weight = 0.01;
  Eigen::MatrixXd state_weight =
      Eigen::MatrixXd::Identity(state_num, state_num);
  Eigen::MatrixXd pred_state_weight = Eigen::MatrixXd::Identity(
      pred_horizon * state_num, pred_horizon * state_num);
  for (int i = 0; i < pred_horizon; i++) {
    pred_state_weight.block(i * state_num + 0, i * state_num + 0, 1, 1) *= 30.0;
    pred_state_weight.block(i * state_num + 1, i * state_num + 1, 1, 1) *= 30.0;
    pred_state_weight.block(i * state_num + 2, i * state_num + 2, 1, 1) *= 30.0;
    pred_state_weight.block(i * state_num + 3, i * state_num + 3, 1, 1) *= 0.1;
    pred_state_weight.block(i * state_num + 4, i * state_num + 4, 1, 1) *= 0.1;
  }
  Eigen::MatrixXd pred_input_weight =
      Eigen::MatrixXd::Identity(ctl_horizon * ctl_num, ctl_horizon * ctl_num) *
      input_weight;
  Eigen::VectorXd pred_input = Eigen::VectorXd::Zero(ctl_horizon * ctl_num);

  for (int i = 0; i < 40; i++) {
    Eigen::VectorXd x_ref =
        generate_reference_trajectory(pred_horizon, dt, state_num, global_time);
    Eigen::VectorXd ref_point = getReference(path, state, 0.6);
    // compute predicted Matrix
    std::vector<Model::model> linearizedModels;
    Eigen::VectorXd pred_ref_points (pred_horizon * state_num);
    for (int k = 0; k < pred_horizon; ++k) {
      linearizedModels.push_back(robot_model.getLinearizedModel(
        ref_point));
      pred_ref_points.segment(k*state_num, state_num) = ref_point;
    }
    predictor.computePredictionMatrix(linearizedModels);
    Eigen::MatrixXd pred_trans_matrix = predictor.getPredTransMatrix();
    Eigen::MatrixXd pred_input_matrix = predictor.getPredInputMatrix();
    Eigen::VectorXd error = pred_trans_matrix * state - pred_ref_points;

    if (!opt.computePredInput(error, pred_trans_matrix, pred_state_weight,
                              pred_input_matrix, pred_input_weight, pred_input))
      pred_input = 0.0 * pred_input;
    robot_model.update(pred_input.segment(0, ctl_num), dt);
    state = robot_model.getState();
    global_time += dt;
    std::cout << "===== Time: " << global_time << " s =====" << std::endl;
    std::cout << "Current State:\n" << state.transpose() << std::endl;
    std::cout << "Reference State (1st step):\n"
              << ref_point.transpose() << std::endl;
    std::cout << "--------------------------------\n" << std::endl;
  }

  return 0;
}