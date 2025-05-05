#include "model.hpp"
#include "utils.hpp"

Model::Model(int state_num, int ctl_num)
:state_num_(state_num), ctl_num_(ctl_num)
{
    transition_matrix_.resize(state_num, state_num);
    state_.resize(state_num,1);
    input_matrix_.resize(state_num, ctl_num);
    input_.resize(ctl_num, 1);
}
Eigen::MatrixXd Model::getTransMatrix()
{
    return transition_matrix_;
}
Eigen::MatrixXd Model::getInputMatrix()
{
    return input_matrix_;
}
Eigen::VectorXd Model::getState()
{
    return state_;
}
void Model::update()
{
    Eigen::VectorXd next_state;
    next_state = transition_matrix_ * state_ + input_matrix_ * input_;

}

void Model::setModel(Eigen::MatrixXd transition_matrix, Eigen::MatrixXd input_matrix)
{
    transition_matrix_ = transition_matrix;
    input_matrix_ = input_matrix;
}

void Model::discretizeModel(int n, double dt)
{
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(transition_matrix_.rows(),transition_matrix_.cols());

    Eigen::MatrixXd exp = I;
    for(int i = 1; i <= n; i++)
    {
        exp += powMatrix(transition_matrix_, i)/factorial(i);
    }

    Eigen::MatrixXd discrete_transition_matrix = exp;

    Eigen::MatrixXd discrete_input_matrix = Eigen::MatrixXd(input_matrix_.rows(), input_matrix_.cols());
    discrete_input_matrix *= 0;
    
    for(int i = 1; i<= n; i++)
    {
        discrete_input_matrix += dt*i/factorial(i)*powMatrix(transition_matrix_,i-1)*input_matrix_;
    }

    transition_matrix_ = discrete_transition_matrix;
    input_matrix_ = discrete_input_matrix;   
}