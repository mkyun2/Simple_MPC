#include <Eigen/Dense>

class Model {
public:
  Model(int state_num, int ctl_num);
  struct model {
    Eigen::MatrixXd transition_matrix;
    Eigen::MatrixXd input_matrix;
    Eigen::VectorXd uncertainties_term;
  };
  model getLinearizedModel(const Eigen::VectorXd &ref);
  model getDiscretizedModel(const model &model_info, int n, double dt);
  void init(Eigen::VectorXd init_state);
  void applyControlInput(Eigen::VectorXd input);
  void update(Eigen::VectorXd input, double dt);
  Eigen::MatrixXd getTransMatrix();
  Eigen::MatrixXd getInputMatrix();
  Eigen::VectorXd getState();

private:
  int state_num_;
  int ctl_num_;

  Eigen::MatrixXd transition_matrix_;
  Eigen::MatrixXd input_matrix_;

  Eigen::VectorXd state_;
  Eigen::VectorXd input_;
  void setModel(Eigen::MatrixXd transition_matrix,
                Eigen::MatrixXd input_matrix);
};
