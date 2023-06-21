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

Eigen::Matrix3d RPY_to_SO3(const Eigen::Vector3d& rpy)
{
  double roll = rpy(0);
  double pitch = rpy(1);
  double yaw = rpy(2);

  Eigen::AngleAxisd roll_rotation(roll, Eigen::Vector3d::UnitX());
  Eigen::AngleAxisd pitch_rotation(pitch, Eigen::Vector3d::UnitY());
  Eigen::AngleAxisd yaw_rotation(yaw, Eigen::Vector3d::UnitZ());

  Eigen::Matrix3d R = yaw_rotation.matrix() * pitch_rotation.matrix() * roll_rotation.matrix();

  return R;
}

Eigen::Vector3d SO3_to_RPY(const Eigen::Matrix3d& R)
{
  Eigen::Vector3d rpy;

  double roll, pitch, yaw;

  pitch = asin(-R(2, 0));

  if (fabs(pitch - M_PI / 2.0) < 1e-6)
  {
    roll = 0;
    yaw = atan2(R(0, 1), R(0, 2));
  }
  else if (fabs(pitch + M_PI / 2.0) < 1e-6)
  {
    roll = 0;
    yaw = atan2(-R(0, 1), -R(0, 2));
  }
  else
  {
    roll = atan2(R(2, 1), R(2, 2));
    yaw = atan2(R(1, 0), R(0, 0));
  }

  rpy << roll, pitch, yaw;

  return rpy;
}

// Ordering is wxyz
Eigen::Matrix3d quat_to_SO3(const Eigen::Vector4d& q)
{
  Eigen::Matrix3d R;
  double q0 = q(0);
  double q1 = q(1);
  double q2 = q(2);
  double q3 = q(3);

  R << q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2),
       2 * (q1 * q2 + q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3, 2 * (q2 * q3 - q0 * q1),
       2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3;

  return R;
}

Eigen::Vector4d SO3_to_quat(const Eigen::Matrix3d& R)
{
  Eigen::Vector4d q;
  double trace = R.trace();
  double r11 = R(0, 0);
  double r12 = R(0, 1);
  double r13 = R(0, 2);
  double r21 = R(1, 0);
  double r22 = R(1, 1);
  double r23 = R(1, 2);
  double r31 = R(2, 0);
  double r32 = R(2, 1);
  double r33 = R(2, 2);

  if (trace > 0)
  {
    double s = 0.5 / sqrt(trace + 1.0);
    q(0) = 0.25 / s;
    q(1) = (r32 - r23) * s;
    q(2) = (r13 - r31) * s;
    q(3) = (r21 - r12) * s;
  }
  else if (r11 > r22 && r11 > r33)
  {
    double s = 2.0 * sqrt(1.0 + r11 - r22 - r33);
    q(0) = (r32 - r23) / s;
    q(1) = 0.25 * s;
    q(2) = (r12 + r21) / s;
    q(3) = (r13 + r31) / s;
  }
  else if (r22 > r33)
  {
    double s = 2.0 * sqrt(1.0 + r22 - r11 - r33);
    q(0) = (r13 - r31) / s;
    q(1) = (r12 + r21) / s;
    q(2) = 0.25 * s;
    q(3) = (r23 + r32) / s;
  }
  else
  {
    double s = 2.0 * sqrt(1.0 + r33 - r11 - r22);
    q(0) = (r21 - r12) / s;
    q(1) = (r13 + r31) / s;
    q(2) = (r23 + r32) / s;
    q(3) = 0.25 * s;
  }

  return q;
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

inline Eigen::Matrix3d SE3_left_jacobian_Q_matrix(const Vector6d& se3)
{
  Eigen::Vector3d rho(se3[3], se3[4], se3[5]);
  Eigen::Vector3d phi(se3[0], se3[1], se3[2]);

  Eigen::Matrix3d rx = skew(rho);
  Eigen::Matrix3d px = skew(phi);

  double ph = phi.norm();
  double ph2 = ph * ph;
  double ph3 = ph2 * ph;
  double ph4 = ph3 * ph;
  double ph5 = ph4 * ph;

  double cph = cos(ph);
  double sph = sin(ph);

  double m1 = 0.5;
  double m2 = (ph - sph) / ph3;
  double m3 = (0.5 * ph2 + cph - 1.) / ph4;
  double m4 = (ph - 1.5 * sph + 0.5 * ph * cph) / ph5;

  Eigen::Matrix3d t1 = rx;
  Eigen::Matrix3d t2 = px * rx + rx * px + (px * rx) * px;
  Eigen::Matrix3d t3 = (px * px) * rx + (rx * px) * px - (3.0 * (px * rx) * px);
  Eigen::Matrix3d t4 = ((px * rx) * px) * px + ((px * px) * rx) * px;

  return m1 * t1 + m2 * t2 + m3 * t3 + m4 * t4;
}

inline Eigen::Matrix<double, 6, 6> SE3_left_jacobian(const Vector6d& se3)
{
  Eigen::Vector3d omega(se3[3], se3[4], se3[5]);
  Eigen::Vector3d v(se3[0], se3[1], se3[2]);

  Eigen::Matrix3d so3_left_jac = SO3_left_jacobian(omega);
  Eigen::Matrix3d Q = SE3_left_jacobian_Q_matrix(se3);

  Eigen::Matrix<double, 6, 6> J;
  J.block<3, 3>(0, 0) = so3_left_jac;
  J.block<3, 3>(3, 3) = so3_left_jac;
  J.block<3, 3>(0, 3) = Q;
  J.block<3, 3>(3, 0) = Eigen::Matrix3d::Zero();
  return J;
}

inline Eigen::Matrix<double, 6, 6> SE3_inverse_left_jacobian(const Vector6d& se3)
{
  Eigen::Vector3d omega(se3[3], se3[4], se3[5]);
  Eigen::Vector3d v(se3[0], se3[1], se3[2]);

  Eigen::Matrix3d so3_inv_left_jac = SO3_inverse_left_jacobian(omega);
  Eigen::Matrix3d Q = SE3_left_jacobian_Q_matrix(se3);

  Eigen::Matrix<double, 6, 6> J_inv;
  J_inv.block<3, 3>(0, 0) = so3_inv_left_jac;
  J_inv.block<3, 3>(3, 3) = so3_inv_left_jac;
  J_inv.block<3, 3>(0, 3) = -(so3_inv_left_jac * Q) * so3_inv_left_jac;
  J_inv.block<3, 3>(3, 0) = Eigen::Matrix3d::Zero();
  return J_inv;
}

#endif // NANOLIE_H
