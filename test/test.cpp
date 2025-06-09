#include <gtest/gtest.h>

#include <iostream>

#include "mpc.hpp"


TEST(MPC, BasicAssertions) {
  EXPECT_STRNE("hello", "world");
  // Expect equality.
  EXPECT_EQ(7 * 6, 42);

  Eigen::MatrixXd matrix = Eigen::MatrixXd(3, 3);
  matrix << 4, 12, -16,
            12, 37, -43,
            -16, -43, 98;
  Optimizer opt;
  bool result = opt.cholesky_decomposition(matrix);
  EXPECT_EQ(result, true);
}