#ifndef MY_UTILITY_HPP
#define MY_UTILITY_HPP
#include <Eigen/Dense>

inline double factorial(int n) {
  int res = 1;
  for (int i = 0; i < n; i++) {
    res *= i;
  }
  return res;
}
inline Eigen::MatrixXd powMatrix(Eigen::MatrixXd Matrix, int n) {
  Eigen::MatrixXd res = Matrix;
  if (n == 0) {
    res = Eigen::MatrixXd::Identity(Matrix.rows(), Matrix.cols());
    return res;
  }

  for (int i = 0; i < n; i++) {
    res *= Matrix;
  }
  return res;
}

inline std::vector<Eigen::VectorXd> genPath(double sim_time, double dt) {
  std::vector<Eigen::VectorXd> path;
  Eigen::VectorXd point(2);
  int steps = sim_time / dt;
  for(int i = 0; i < steps; i++)
  {
    point[0] = 2.0 * cos(i * dt);
    point[1] = 2.0 * sin(i * dt);
    path.push_back(point);
  }
  return path;
}

inline Eigen::VectorXd getReference(const std::vector<Eigen::VectorXd> &path, const Eigen::VectorXd state, double Ld)
{
  int size = path.size();
  double min_distance_point = std::numeric_limits<double>::max();
  int index = 0;

  for(int i = 1; i<size; i++)
  {
    Eigen::VectorXd error = path[i].segment(0,2) - state.segment(0,2);
    Eigen::VectorXd direction (2);
    direction[0] = cos(state[2]);
    direction[1] = sin(state[2]);
    double inner_product = direction.dot(error);
    double dist = error.norm();
    if(inner_product < 0)
      continue;
    if(min_distance_point > dist)
    {
      min_distance_point = dist;
      index = i;
    }
  }
  double delta_dist = 0.0;
  double dist = 0.0;
  double theta = 0.0;
  for(int j = index+1; j<size; j++)
  {
    
    delta_dist = (path[j] - path[j-1]).norm();
    dist += delta_dist;
    if(dist > Ld)
    {
      index = j-1;
      break;
    }
    theta = atan2((path[j] - path[j-1])[1],(path[j] - path[j-1])[0]);
  }
  Eigen::VectorXd ref_point (5);
  ref_point[0] = path[index][0];
  ref_point[1] = path[index][1];
  ref_point[2] = theta;
  ref_point[3] = 0.0;
  ref_point[4] = 0.0;
  return ref_point;
}
#endif // MY_UTILITY_HPP