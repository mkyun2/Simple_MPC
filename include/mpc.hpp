#include <Eigen/Dense>
class Predictor{
    
public:
    Predictor(int state_num, int ctl_num, int pred_horizon, int ctl_horizon);
    void computePredictionMatrix(Eigen::MatrixXd transition_matrix, Eigen::MatrixXd input_matrix);
    Eigen::MatrixXd getPredTransMatrix();
    Eigen::MatrixXd getPredInputMatrix();
    

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

class Optimizer{
    
public:
    Optimizer();
    void setConstraintMatrix(Eigen::MatrixXd E, Eigen::VectorXd b);
    void computePredInput(Eigen::MatrixXd pred_state, Eigen::MatrixXd pred_state_matrix, Eigen::MatrixXd pred_state_weight, Eigen::MatrixXd pred_input_matrix, Eigen::MatrixXd pred_input_weight, Eigen::VectorXd &pred_input);
    bool cholesky_decomposition(Eigen::MatrixXd matrix);
private:

   Eigen::MatrixXd coeff_;
   Eigen::VectorXd b_;
};