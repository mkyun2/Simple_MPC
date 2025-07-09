#ifndef MY_UTILITY_HPP
#define MY_UTILITY_HPP
#include <Eigen/Dense>

inline double factorial(int n) {
  int res = 1;
  for (int i = n; i > 0; i--) {
    res *= i;
  }
  return res;
}
inline Eigen::MatrixXd powMatrix(Eigen::MatrixXd Matrix, int n) {
  Eigen::MatrixXd res = Eigen::MatrixXd::Identity(Matrix.rows(), Matrix.cols());;
  if (n == 0) {
    return res;
  }

  for (int i = 0; i < n; i++) {
    res *= Matrix;
  }
  return res;
}

inline std::vector<Eigen::VectorXd> genPath(std::string &path_type) {
  std::vector<Eigen::VectorXd> path;
  Eigen::VectorXd point(3);
  double path_length = 2.0 * M_PI;
  ;
  double dt = 0.05;
  int steps = path_length / dt;

  if (path_type == "circle") {
    for (int i = 0; i < steps; i++) {
      double t = i * dt;
      point[0] = 2.0 * cos(t);
      point[1] = 2.0 * sin(t);

      double dx_dt = -2.0 * sin(t);
      double dy_dt = 2.0 * cos(t);
      double ref_theta = atan2(dy_dt, dx_dt);

      point[2] = ref_theta;
      path.push_back(point);
    }
  } else if (path_type == "hallway") {
    for (int i = 0; i < steps; i++) {
      double t = i * dt;
      double dx_dt, dy_dt;
      if (i < steps / 2) {
        point[1] = t;
        dx_dt = 0.0001;
        dy_dt = 1;
        point[2] = atan2(dy_dt, dx_dt);
      } else {
        point[0] = t - point[1];
        dx_dt = 1;
        dy_dt = 0.0001;
        point[2] = atan2(dy_dt, dx_dt);
      }
      path.push_back(point);
    }

  } else if (path_type == "sin") {
    for (int i = 0; i < steps; i++) {
      double t = i * dt;
      double dx_dt, dy_dt;
      point[0] = t;
      point[1] = sin(t);

      dx_dt = 1;
      dy_dt = cos(t);
      point[2] = atan2(dy_dt, dx_dt);
      path.push_back(point);
    }
  }
  return path;
}

inline Eigen::VectorXd getReference(const std::vector<Eigen::VectorXd> &path,
                                    const Eigen::VectorXd &state,
                                    int sample_num) {
  int size = path.size();
  double min_distance_point = std::numeric_limits<double>::max();
  int index_min = 0;
  for (int i = 1; i < size; i++) {
    Eigen::VectorXd error = path[i].segment(0, 2) - state.segment(0, 2);
    Eigen::VectorXd direction(2);
    direction[0] = cos(state[2]);
    direction[1] = sin(state[2]);
    double inner_product = direction.dot(error);
    double dist = error.norm();

    if (min_distance_point > dist) {
      min_distance_point = dist;
      index_min = i;
    }
  }

  int index_ref = index_min;
  Eigen::VectorXd ref_point(sample_num * state.rows());

  for (int j = 0; j < sample_num; j++) {
    ref_point[j * state.rows() + 0] = path[index_ref][0];
    ref_point[j * state.rows() + 1] = path[index_ref][1];
    ref_point[j * state.rows() + 2] = path[index_ref][2];
    // atan2(path[index_Ld][1] - path[index_Ld-1][1],
    // path[index_Ld][0] - path[index_Ld-1][0]);
    ref_point[j * state.rows() + 3] = 0.0; // 1.0;
    ref_point[j * state.rows() + 4] =
        0.0; // atan2(sin(ref_point[2]),cos(ref_point[2]))/M_PI * 1.5;

    if (index_ref < (size - 1))
      index_ref++;
  }

  return ref_point;
}
#endif // MY_UTILITY_HPP