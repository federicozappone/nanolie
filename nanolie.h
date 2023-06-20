#ifndef NANOLIE_H
#define NANOLIE_H

#include <Eigen/Core>
#include <Eigen/Geometry>

typedef Eigen::Matrix<double, 6, 1> Vector6d;

inline Eigen::Matrix3d skew(const Eigen::Vector3d& vec)
{
  return (Eigen::Matrix3d() << 0.0, -vec[2], vec[1], vec[2], 0.0,
          -vec[0], -vec[1], vec[0], 0.0).finished();
}

inline Eigen::Vector3d SO3_wedge(const Eigen::Matrix3d& Omega)
{
  Eigen::Vector3d omega;
  omega << -Omega(1, 2), Omega(0, 2), -Omega(0, 1);
  return omega;
}

inline Eigen::Vector3d SO3_vee(const Eigen::Matrix3d& Omega)
{
  Eigen::Vector3d omega;
  omega << Omega(2, 1), Omega(0, 2), Omega(1, 0);
  return omega;
}

inline Eigen::Matrix3d SO3_exp(const Eigen::Vector3d& omega)
{
  Eigen::Matrix3d Omega = skew(omega);
  double theta = omega.norm();
  Eigen::Matrix3d R;

  if (theta < std::numeric_limits<double>::epsilon())
  {
    R = Eigen::Matrix3d::Identity() + Omega;
  }
  else
  {
    R = Eigen::Matrix3d::Identity() + (std::sin(theta) / theta) * Omega +
        ((1 - std::cos(theta)) / (theta * theta)) * Omega * Omega;
  }

  return R;
}

inline Eigen::Vector3d SO3_log(const Eigen::Matrix3d& R)
{
  Eigen::Vector3d omega;

  double cos_theta = (R.trace() - 1) / 2.0;
  if (cos_theta > 1.0)
    cos_theta = 1.0;
  else if (cos_theta < -1.0)
    cos_theta = -1.0;

  double theta = std::acos(cos_theta);
  double sin_theta = std::sin(theta);

  if (std::abs(sin_theta) < std::numeric_limits<double>::epsilon())
  {
    omega = 0.5 * SO3_wedge(R - Eigen::Matrix3d::Identity());
  }
  else
  {
    omega = theta / (2.0 * sin_theta) * SO3_wedge(R - R.transpose());
  }

  return omega;
}

inline Eigen::Matrix3d SO3_inverse(const Eigen::Matrix3d& Omega)
{
  return Omega.transpose();
}

inline Eigen::Matrix3d SO3_left_jacobian(const Eigen::Vector3d& omega)
{
  Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
  double theta = omega.norm();

  if (theta < std::numeric_limits<double>::epsilon())
  {
    J = Eigen::Matrix3d::Identity() - 0.5 * skew(omega);
  }
  else
  {
    Eigen::Matrix3d Omega = skew(omega);
    J = Eigen::Matrix3d::Identity() + (1 - std::cos(theta)) / (theta * theta) * Omega +
        (theta - std::sin(theta)) / (theta * theta * theta) * Omega * Omega;
  }

  return J;
}

inline Eigen::Matrix3d SO3_inverse_left_jacobian(const Eigen::Vector3d& omega)
{
  Eigen::Matrix3d J_inv = Eigen::Matrix3d::Zero();
  double theta = omega.norm();

  if (theta < std::numeric_limits<double>::epsilon())
  {
    J_inv = Eigen::Matrix3d::Identity() + 0.5 * skew(omega);
  }
  else
  {
    Eigen::Matrix3d Omega = skew(omega);
    J_inv = Eigen::Matrix3d::Identity() -
            (0.5 * theta - std::sin(0.5 * theta)) / (theta * theta * std::sin(theta)) * Omega +
            (theta - std::cos(theta)) / (theta * theta * std::sin(theta)) * Omega * Omega;
  }

  return J_inv;
}


inline Eigen::Matrix4d SE3_wedge(const Eigen::VectorXd& xi)
{
  Eigen::Matrix4d Omega;
  Omega << 0.0, -xi(2), xi(1), xi(3),
           xi(2), 0.0, -xi(0), xi(4),
           -xi(1), xi(0), 0.0, xi(5),
           0.0, 0.0, 0.0, 0.0;
  return Omega;
}

inline Eigen::VectorXd SE3_vee(const Eigen::Matrix4d& T)
{
  Eigen::VectorXd xi(6);
  xi << T(2, 1), T(0, 2), T(1, 0), T(0, 3), T(1, 3), T(2, 3);
  return xi;
}

inline Eigen::Isometry3d SE3_exp(const Vector6d& se3)
{
  Eigen::Vector3d omega(se3[3], se3[4], se3[5]);
  Eigen::Vector3d v(se3[0], se3[1], se3[2]);

  double theta = omega.norm();
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d p = Eigen::Vector3d::Zero();

  if (theta < std::numeric_limits<double>::epsilon())
  {
    R = Eigen::Matrix3d::Zero();
    p = v;
  }
  else
  {
    R = SO3_exp(omega);
    p = SO3_left_jacobian(omega) * v;
  }

  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = R;
  T.translation() = p;

  return T;
}

inline Vector6d SE3_log(const Eigen::Isometry3d& T)
{
  Eigen::Matrix3d R = T.linear();
  Eigen::Vector3d p = T.translation();

  Eigen::Vector3d omega;

  omega = SO3_log(R);
  Eigen::Vector3d v = SO3_inverse_left_jacobian(omega) * p;

  Vector6d se3;
  se3 << v[0], v[1], v[2], omega[0], omega[1], omega[2];

  return se3;
}

inline Eigen::Isometry3d SE3_inverse(const Eigen::Isometry3d& T)
{
  Eigen::Isometry3d T_inv;
  T_inv.linear() = SO3_inverse(T.linear());
  T_inv.translation() = -SO3_inverse(T.linear()) * T.translation();
  return T_inv;
}

inline Eigen::Matrix<double, 6, 6> SE3_adjoint(const Eigen::Isometry3d& T)
{
  Eigen::Matrix<double, 6, 6> adj;
  adj.block<3, 3>(0, 0) = T.linear();
  adj.block<3, 3>(3, 3) = T.linear();
  adj.block<3, 3>(3, 0) = Eigen::Matrix3d::Zero();
  adj.block<3, 3>(0, 3) = T.linear() * skew(T.translation());
  return adj;
}

inline Eigen::Matrix<double, 6, 6> SE3_dual_adjoint(const Eigen::Isometry3d& T)
{
  Eigen::Matrix<double, 6, 6> dual_adj;
  dual_adj.block<3, 3>(0, 0) = T.linear().transpose();
  dual_adj.block<3, 3>(3, 3) = T.linear().transpose();
  dual_adj.block<3, 3>(3, 0) = Eigen::Matrix3d::Zero();
  dual_adj.block<3, 3>(0, 3) = skew(T.translation()) * T.linear().transpose();
  return dual_adj;
}

inline Eigen::MatrixXd SE3_left_jacobian(const Eigen::Vector3d& rho)
{
  Eigen::MatrixXd J(6, 6);
  J.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
  J.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
  J.block<3, 3>(0, 3) = skew(rho);
  J.block<3, 3>(3, 0) = Eigen::Matrix3d::Zero();
  return J;
}

inline Eigen::MatrixXd SE3_inverse_left_jacobian(const Eigen::Vector3d& rho)
{
  Eigen::MatrixXd J_inv(6, 6);
  J_inv.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
  J_inv.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
  J_inv.block<3, 3>(0, 3) = -skew(rho);
  J_inv.block<3, 3>(3, 0) = Eigen::Matrix3d::Zero();
  return J_inv;
}

#endif // NANOLIE_H
