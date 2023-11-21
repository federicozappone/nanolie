#include <iostream>
#include <gtest/gtest.h>
#include "so3.h"

using namespace std;
using namespace Eigen;
using namespace nanolie;

TEST(SO3Test, SO3WedgeVeeTest)
{
  Vector3d omega(0.1, 0.2, 0.3);
  ASSERT_TRUE(SO3::vee(SO3::wedge(omega)).isApprox(omega));
}

TEST(SO3Test, SO3RPYTest)
{
  Vector3d rpy(0.1, 0.2, 0.3);
  ASSERT_TRUE(SO3::from_rpy(rpy).to_rpy().isApprox(rpy));
}

TEST(SO3Test, SO3QuaternionTest)
{
  Vector4d q1(1.0, 0.0, 0.0, 0.0);
  Vector4d q2(0.0, 1.0, 0.0, 0.0);
  Vector4d q3(0.0, 0.0, 1.0, 0.0);
  Vector4d q4(0.0, 0.0, 0.0, 1.0);

  ASSERT_TRUE(SO3::from_quat(q1).to_quat().isApprox(q1));
  ASSERT_TRUE(SO3::from_quat(q2).to_quat().isApprox(q2));
  ASSERT_TRUE(SO3::from_quat(q3).to_quat().isApprox(q3));
  ASSERT_TRUE(SO3::from_quat(q4).to_quat().isApprox(q4));
}

TEST(SO3Test, SO3ExpLogTest)
{
  Vector3d omega(0.1, 0.15, 0.02);
  ASSERT_TRUE(SO3::exp(omega).log().isApprox(omega));
}

TEST(SO3Test, SO3ExpZeroTest)
{
  Vector3d omega(0.0, 0.0, 0.0);
  ASSERT_TRUE(SO3::exp(omega).matrix().isApprox(Matrix3d::Identity()));
}

TEST(SO3Test, SO3InverseTest)
{
  Vector3d omega(0.1, 0.15, 0.02);
  SO3 rot = SO3::exp(omega);
  ASSERT_TRUE((rot.inverse() * rot).matrix().isApprox(Matrix3d::Identity()));
}

TEST(SO3Test, SO3LeftJacobianTest)
{
  Vector3d omega(0.1, 0.15, 0.02);
  ASSERT_TRUE((SO3::left_jacobian(omega) * SO3::inverse_left_jacobian(omega))
                  .isApprox(Matrix3d::Identity()));
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
