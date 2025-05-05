#include <Eigen/Dense>

double factorial(int n)
{
    int res = 1;
    for(int i = 0; i < n; i++)
    {
        res*=i;
    }
    return res;
}
Eigen::MatrixXd powMatrix(Eigen::MatrixXd Matrix, int n)
{
    Eigen::MatrixXd res = Matrix;
    if(n==0)
    {
        res = Eigen::MatrixXd::Identity(Matrix.rows(),Matrix.cols());
        return res;
    }

    for(int i = 0; i < n; i++)
    {
        res *= Matrix;
    }
    return res;
}