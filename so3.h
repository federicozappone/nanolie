#ifndef SO3_H
#define SO3_H

#include <Eigen/Core>
#include <Eigen/Geometry>


class SO3
{
public:
  SO3() : _rot(Eigen::Matrix3d::Identity()) { }
  SO3(const SO3& lhs) : _rot(lhs._rot) { }

  static Eigen::Matrix3d wedge(const Eigen::Vector3d& omega)
  {
    return (Eigen::Matrix3d() << 0.0, -omega[2], omega[1], omega[2], 0.0, -omega[0], -omega[1],
            omega[0], 0.0)
        .finished();
  }

  static Eigen::Vector3d vee(const Eigen::Matrix3d& Omega)
  {
    return (Eigen::Vector3d() << Omega(2, 1), Omega(0, 2), Omega(1, 0)).finished();
  }

  static SO3 exp(const Eigen::Vector3d& omega)
  {
    Eigen::Matrix3d Omega = SO3::wedge(omega);
    double theta = omega.norm();
    SO3 ret;

    if (theta < std::numeric_limits<double>::epsilon())
    {
      ret._rot = Eigen::Matrix3d::Identity() + Omega;
    }
    else
    {
      ret._rot = Eigen::Matrix3d::Identity() + (std::sin(theta) / theta) * Omega +
                 ((1 - std::cos(theta)) / (theta * theta)) * Omega * Omega;
    }

    return ret;
  }

  Eigen::Vector3d log() const
  {
    Eigen::Vector3d omega;

    double cos_theta = 0.5 * (_rot.trace()) - 0.5;
    cos_theta = std::clamp(cos_theta, -1.0, 1.0);

    double theta = std::acos(cos_theta);
    double sin_theta = std::sin(theta);

    if (std::abs(sin_theta) < std::numeric_limits<double>::epsilon())
    {
      omega = 0.5 * vee(_rot - Eigen::Matrix3d::Identity());
    }
    else
    {
      omega = theta / (2.0 * sin_theta) * vee(_rot - _rot.transpose());
    }

    return omega;
  }

  SO3 inverse() const
  {
    SO3 ret;
    ret._rot = _rot.transpose();
    return ret;
  }

  static SO3 from_rpy(const Eigen::Vector3d& rpy)
  {
    SO3 ret;

    double roll = rpy(0);
    double pitch = rpy(1);
    double yaw = rpy(2);

    Eigen::AngleAxisd roll_rotation(roll, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitch_rotation(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yaw_rotation(yaw, Eigen::Vector3d::UnitZ());

    ret._rot = yaw_rotation.matrix() * pitch_rotation.matrix() * roll_rotation.matrix();

    return ret;
  }

  Eigen::Vector3d to_rpy() const
  {
    Eigen::Vector3d rpy;

    double roll, pitch, yaw;

    pitch = asin(-_rot(2, 0));

    if (fabs(pitch - M_PI / 2.0) < 1e-6)
    {
      roll = 0;
      yaw = atan2(_rot(0, 1), _rot(0, 2));
    }
    else if (fabs(pitch + M_PI / 2.0) < 1e-6)
    {
      roll = 0;
      yaw = atan2(-_rot(0, 1), -_rot(0, 2));
    }
    else
    {
      roll = atan2(_rot(2, 1), _rot(2, 2));
      yaw = atan2(_rot(1, 0), _rot(0, 0));
    }

    rpy << roll, pitch, yaw;

    return rpy;
  }

  // Ordering is wxyz
  static SO3 from_quat(const Eigen::Vector4d& q)
  {
    SO3 ret;
    double qw = q(0);
    double qx = q(1);
    double qy = q(2);
    double qz = q(3);

    double qx2 = qx * qx;
    double qy2 = qy * qy;
    double qz2 = qz * qz;

    double r11 = 1.0 - 2.0 * (qy2 + qz2);
    double r12 = 2.0 * (qx * qy - qw * qz);
    double r13 = 2.0 * (qw * qy + qx * qz);

    double r21 = 2.0 * (qw * qz + qx * qy);
    double r22 = 1.0 - 2.0 * (qx2 + qz2);
    double r23 = 2.0 * (qy * qz - qw * qx);

    double r31 = 2.0 * (qx * qz - qw * qy);
    double r32 = 2.0 * (qw * qx + qy * qz);
    double r33 = 1.0 - 2.0 * (qx2 + qy2);

    ret._rot << r11, r12, r13, r21, r22, r23, r31, r32, r33;

    return ret;
  }

  // Ordering is wxyz
  Eigen::Vector4d to_quat() const
  {
    Eigen::Vector4d q;
    double r11 = _rot(0, 0);
    double r12 = _rot(0, 1);
    double r13 = _rot(0, 2);
    double r21 = _rot(1, 0);
    double r22 = _rot(1, 1);
    double r23 = _rot(1, 2);
    double r31 = _rot(2, 0);
    double r32 = _rot(2, 1);
    double r33 = _rot(2, 2);

    double qw = 0.5 * std::sqrt(1.0 + r11 + r22 + r33);

    if (qw < std::numeric_limits<double>::epsilon())
    {
      if (r11 > r22 && r11 > r33)
      {
        double d = 2.0 * std::sqrt(1.0 + r11 - r22 - r33);
        q(0) = (r32 - r23) / d;
        q(1) = 0.25 * d;
        q(2) = (r21 + r12) / d;
        q(3) = (r13 + r31) / d;
      }
      else if (r22 > r33)
      {
        double d = 2.0 * std::sqrt(1.0 + r22 - r11 - r33);
        q(0) = (r13 - r31) / d;
        q(1) = (r21 + r12) / d;
        q(2) = 0.25 * d;
        q(3) = (r32 + r23) / d;
      }
      else
      {
        double d = 2.0 * std::sqrt(1.0 + r33 - r11 - r22);
        q(0) = (r21 - r12) / d;
        q(1) = (r13 + r31) / d;
        q(2) = (r32 + r23) / d;
        q(3) = 0.25 * d;
      }
    }
    else
    {
      double d = 4.0 * qw;
      q(0) = 0.5 * std::sqrt(1.0 + r11 + r22 + r33);
      q(1) = (r32 - r23) / d;
      q(2) = (r13 - r31) / d;
      q(3) = (r21 - r12) / d;
    }

    return q;
  }

  static Eigen::Matrix3d left_jacobian(const Eigen::Vector3d& omega)
  {
    Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
    double theta = omega.norm();

    if (theta < std::numeric_limits<double>::epsilon())
    {
      J = Eigen::Matrix3d::Identity() - 0.5 * wedge(omega);
    }
    else
    {
        Eigen::Vector3d axis = omega / theta;
        double s = std::sin(theta);
        double c = std::cos(theta);

        J = (s / theta) * Eigen::Matrix3d::Identity() +
            (1.0 - s / theta) * (axis * axis.transpose()) +
            ((1.0 - c) / theta) * SO3::wedge(axis);
    }

    return J;
  }

  static Eigen::Matrix3d inverse_left_jacobian(const Eigen::Vector3d& omega)
  {
    Eigen::Matrix3d J_inv = Eigen::Matrix3d::Zero();
    double theta = omega.norm();

    if (theta < std::numeric_limits<double>::epsilon())
    {
      J_inv = Eigen::Matrix3d::Identity() + 0.5 * wedge(omega);
    }
    else
    {
      Eigen::Vector3d axis = omega / theta;
      double half_angle = 0.5 * theta;
      double cot_half_angle = 1.0 / std::tan(half_angle);

      J_inv = half_angle * cot_half_angle * Eigen::Matrix3d::Identity() +
            (1.0 - half_angle * cot_half_angle) * (axis * axis.transpose()) -
            half_angle * SO3::wedge(axis);
    }

    return J_inv;
  }

  Eigen::Matrix3d matrix() const { return _rot; }

  void from_matrix(const Eigen::Matrix3d& rot)
  {
    _rot = rot;
  }

  // operators

  SO3 operator=(const SO3& rhs)
  {
    _rot = rhs._rot;
    return *this;
  }

  SO3 operator*=(const SO3& rhs)
  {
    _rot *= rhs._rot;
    return *this;
  }

  friend SO3 operator*(SO3 lhs, const SO3& rhs) { return lhs *= rhs; }

  friend Eigen::Vector3d operator*(Eigen::Vector3d lhs, const SO3& rhs) { return rhs._rot * lhs; }

private:
  Eigen::Matrix3d _rot;
};

#endif  // SO3_H
