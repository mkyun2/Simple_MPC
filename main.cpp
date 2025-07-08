#include <matplotlibcpp.h>

#include <iostream>

#include "mpc.hpp"
#include "utils.hpp"
namespace plt = matplotlibcpp;
int main() {
  std::vector<double> x, y;
  std::vector<double> sx, sy;
  std::vector<double> err_x, err_y;
  std::vector<double> lin_vel, ang_vel;
  std::vector<double> lin_acc, ang_acc;
  std::vector<double> time_vec;
  int state_num = 5;
  int ctl_num = 2;
  int pred_horizon = 20;
  int ctl_horizon = 5;
  double dt = 0.05;
  double global_time = 0.0;
  // def Reference and Init State
  Model robot_model(state_num, ctl_num);
  Eigen::VectorXd init_state(state_num);
  init_state << 0.0, 0.0, M_PI / 4, 0.0, 0.0;
  std::vector<Eigen::VectorXd> path = genPath(2 * M_PI, 0.05);
  for (auto pt : path) {
    x.push_back(pt[0]);
    y.push_back(pt[1]);
  }

  robot_model.init(init_state);
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
  b_max << 1.5, 3.0;
  b_min << -1.5, -3.0;

  opt.addIneqConstraint(ctl_horizon, ctl_num, b_min, b_max, 5.0);

  Eigen::VectorXd state_weight(state_num);
  state_weight << 10.0, 10.0, 10.0, 0.1, 0.1;
  Eigen::VectorXd final_state_weight(state_num);
  state_weight << 100.0, 100.0, 100.0, 0.1, 0.1;
  Eigen::VectorXd input_weight(ctl_num);
  input_weight << 0.01, 0.01;
  std::vector<Eigen::MatrixXd> weights =
      predictor.buildWeightMatrix(state_weight, input_weight, final_state_weight);

  Eigen::VectorXd pred_input = Eigen::VectorXd::Zero(ctl_horizon * ctl_num);
  for (int i = 0; i < 200; i++) {
    Eigen::VectorXd ref_point = getReference(path, state, pred_horizon);
    // compute predicted Matrix
    std::vector<Model::model> linearizedModels;
    Eigen::VectorXd pred_ref_points(pred_horizon * state_num);
    // Eigen::VectorXd ref_state_k;
    for (int k = 0; k < pred_horizon; ++k) {
      // ref_state_k = x_ref.segment(k * state_num, state_num);
      Model::model continuous_model = robot_model.getLinearizedModel(
          ref_point.segment(k * state_num, state_num));
      Model::model discrete_model =
          robot_model.getDiscretizedModel(continuous_model, 5, dt);
      linearizedModels.push_back(discrete_model);
      pred_ref_points = ref_point;
    }

    Model::model pred_model =
        predictor.computePredictionMatrix(linearizedModels, ref_point);

    Eigen::MatrixXd pred_trans_matrix =
        pred_model.transition_matrix; // predictor.getPredTransMatrix();
    Eigen::MatrixXd pred_input_matrix =
        pred_model.input_matrix; // predictor.getPredInputMatrix();
    Eigen::VectorXd error = (pred_trans_matrix * state) - pred_ref_points;
    for (int k = 0; k < pred_horizon; ++k) {
      double rad_err = error[k * state_num + 2];
      error[k * state_num + 2] = atan2(sin(rad_err), cos(rad_err));
    }
    if (!opt.computePredInput(error, pred_trans_matrix, weights[0],
                              pred_input_matrix, weights[1], pred_input))
      pred_input = 0.0 * pred_input;
    auto pred_state =
        pred_trans_matrix * state + pred_input_matrix * pred_input;
    std::vector<double> pred_x, pred_y;
    std::vector<double> rx, ry;

    err_x.push_back(state[0] - ref_point[0]);
    err_y.push_back(state[1] - ref_point[1]);
    lin_vel.push_back(state[3]);
    ang_vel.push_back(state[4]);
    lin_acc.push_back(pred_input[0]);
    ang_acc.push_back(pred_input[1]);
    time_vec.push_back(global_time);
    for (int p = 0; p < pred_horizon; p++) {
      auto ps = pred_state.segment(p * state_num, state_num);
      pred_x.push_back(ps[0]);
      pred_y.push_back(ps[1]);

      rx.push_back(ref_point.segment(p * state_num, state_num)[0]);
      ry.push_back(ref_point.segment(p * state_num, state_num)[1]);
    }
    robot_model.update(pred_input.segment(0, ctl_num), dt);
    state = robot_model.getState();
    global_time += dt;
    std::cout << "===== Time: " << global_time << " s =====" << std::endl;
    std::cout << "Current State:\n" << state.transpose() << std::endl;
    std::cout << "Reference State (1st step):\n"
              << ref_point.segment(0, state_num).transpose() << std::endl;
    std::cout << "--------------------------------\n" << std::endl;
    sx.push_back(state[0]);
    sy.push_back(state[1]);
    
    plt::clf();
    const long nrows=2, ncols=2;
    plt::subplot(nrows,ncols,1);
    plt::title("Path Following");
    plt::plot(x, y);
    plt::plot(pred_x, pred_y, "b--");
    plt::scatter(rx, ry, 5.0);
    plt::plot(sx, sy, "r--");
    plt::xlim(-3, 3);
    plt::ylim(-3, 3);
    plt::set_aspect(1.0);

    plt::subplot(nrows,ncols,2);
    plt::title("Error");
    plt::named_plot("err_x",time_vec,err_x,"b");
    plt::named_plot("err_y",time_vec,err_y,"r");
    plt::legend();
    plt::xlim(0, 10);
    plt::ylim(-3, 3);
    plt::grid(true);
    plt::set_aspect(1.0);

    plt::subplot(nrows,ncols,3);
    plt::title("Velocity");
    plt::named_plot("lin_vel",time_vec,lin_vel,"r");
    plt::named_plot("ang_vel",time_vec,ang_vel,"b");
    plt::legend();
    plt::xlim(0, 10);
    plt::ylim(-3, 3);
    plt::grid(true);
    plt::set_aspect(1.0);

    plt::subplot(nrows,ncols,4);
    plt::title("Acceleration");
    plt::named_plot("lin_acc",time_vec,lin_acc,"r");
    plt::named_plot("ang_acc",time_vec,ang_acc,"b");
    plt::legend();
    plt::xlim(0, 10);
    plt::ylim(-3, 3);
    plt::grid(true);
    plt::set_aspect(1.0);
    plt::tight_layout();
    plt::pause(0.001);
    
  }

  plt::show();
  return 0;
}