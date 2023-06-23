#include <iostream>
#include <gtest/gtest.h>
#include "se3.h"

using namespace std;
using namespace Eigen;

TEST(SE3Test, SE3WedgeVeeTest)
{
  Vector6d se3(1.0, 2.0, 3.0, 0.1, 0.15, 0.02);
  ASSERT_TRUE(SE3::vee(SE3::wedge(se3)).isApprox(se3));
}

TEST(SE3Test, SE3ExpLogTest)
{
  Vector6d se3(1.0, 2.0, 3.0, 0.1, 0.15, 0.02);
  ASSERT_TRUE(SE3::exp(se3).log().isApprox(se3));
}

TEST(SE3Test, SE3ExpZeroTest)
{
  Vector6d se3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  ASSERT_TRUE(SE3::exp(se3).matrix().isApprox(Matrix4d::Identity()));
}

TEST(SE3Test, SE3InverseTest)
{
  Vector6d se3(1.0, 2.0, 3.0, 0.1, 0.15, 0.02);
  SE3 T = SE3::exp(se3);
  ASSERT_TRUE((T.inverse() * T).matrix().isApprox(Matrix4d::Identity()));
}

TEST(SE3Test, SE3LeftJacobianTest)
{
  Vector6d se3(1.0, 2.0, 3.0, 0.1, 0.15, 0.02);
  ASSERT_TRUE(
      (SE3::left_jacobian(se3) * SE3::inverse_left_jacobian(se3)).isApprox(Matrix6d::Identity()));
}

TEST(SE3Test, SE3MatrixTest)
{
  Vector6d se3(1.0, 2.0, 3.0, 0.1, 0.15, 0.02);
  Matrix4d T = SE3::exp(se3).matrix();

  ASSERT_TRUE(SE3::from_matrix(T).matrix().isApprox(T));
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
