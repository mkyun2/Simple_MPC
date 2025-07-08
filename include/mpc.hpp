#include <Eigen/Dense>

#include "model.hpp"
class Predictor {
public:
  Predictor(int state_num, int ctl_num, int pred_horizon, int ctl_horizon);
  Model::model computePredictionMatrix(std::vector<Model::model> models,
                                       Eigen::VectorXd x_ref);
  Eigen::MatrixXd getPredTransMatrix();
  Eigen::MatrixXd getPredInputMatrix();
  std::vector<Eigen::MatrixXd>
  buildWeightMatrix(Eigen::VectorXd state_weight, Eigen::VectorXd input_weight,
                    Eigen::VectorXd final_state_weight);

private:
  int state_num_;
  int ctl_num_;
  int pred_horizon_;
  int ctl_horizon_;

  Eigen::MatrixXd pred_trans_matrix_;
  Eigen::MatrixXd pred_input_matrix_;
  Eigen::VectorXd pred_state_;
  Eigen::VectorXd pred_input_;
};

class Optimizer {
public:
  Optimizer();
  struct EqualityConstraint {
    Eigen::MatrixXd A_eq;
    Eigen::VectorXd b_eq;
  };

  struct InequalityConstraint {
    Eigen::MatrixXd A_ineq;
    Eigen::VectorXd b_ineq;
    float penalty;
  };
  void addIneqConstraint(int N, int n_u, Eigen::VectorXd u_min,
                         Eigen::VectorXd u_max, float penalty);
  bool computePredInput(Eigen::MatrixXd pred_state,
                        Eigen::MatrixXd pred_state_matrix,
                        Eigen::MatrixXd pred_state_weight,
                        Eigen::MatrixXd pred_input_matrix,
                        Eigen::MatrixXd pred_input_weight,
                        Eigen::VectorXd &pred_input);
  bool cholesky_decomposition(Eigen::MatrixXd matrix);

private:
  // Eigen::MatrixXd KKT_Matrix_;
  Eigen::MatrixXd A_ineq_;
  Eigen::VectorXd b_ineq_;
  std::vector<InequalityConstraint> IneqConstraints_;
};