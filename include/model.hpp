#include <Eigen/Dense>
class Model {
 public:
  Model(int state_num, int ctl_num);
  void applyControlInput(Eigen::VectorXd input);
  void update();
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
  void discretizeModel(int n, double dt);
};
