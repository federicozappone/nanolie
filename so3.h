/**
 * @file so3.h
 * @brief Header file for the SO3 class.
 *
 * @author Federico Zappone
 * @date 2023-07-21
 */
#ifndef SO3_H
#define SO3_H

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace nanolie {

/**
 * @brief Represents a 3D rotation in the Special Orthogonal group SO(3).
 */
class SO3
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

public:
  /**
   * @brief Default constructor. Initializes with the identity rotation.
   */
  SO3() : _rot(Eigen::Matrix3d::Identity()) {}
  /**
   * @brief Copy constructor.
   * @param lhs The SO3 object to copy.
   */
  SO3(const SO3& lhs) : _rot(lhs._rot) {}
  /**
   * @brief Constructor with a rotation matrix.
   * @param rot Rotation matrix.
   */
  SO3(const Eigen::Matrix3d& rot) : _rot(rot) {}

  /**
   * @brief Computes the wedge operator of a 3x1 vector.
   * @param omega 3x1 vector.
   * @return 3x3 matrix representing the wedge operator.
   */
  static Eigen::Matrix3d wedge(const Eigen::Vector3d& omega)
  {
    // clang-format off
    return (Eigen::Matrix3d() << 0.0, -omega[2], omega[1],
                                omega[2], 0.0, -omega[0], 
                                -omega[1], omega[0], 0.0).finished();
    // clang-format on
  }

  /**
   * @brief Computes the vee operator of a 3x3 matrix.
   * @param Omega 3x3 matrix.
   * @return 3x1 vector representing the vee operator.
   */
  static Eigen::Vector3d vee(const Eigen::Matrix3d& Omega)
  {
    return (Eigen::Vector3d() << Omega(2, 1), Omega(0, 2), Omega(1, 0)).finished();
  }

  /**
   * @brief Computes the exponential map for an element in the Lie algebra so(3).
   * @param omega 3x1 vector in so(3).
   * @return Corresponding SO3 element.
   */
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

  /**
   * @brief Computes the logarithmic map for the current SO3 element.
   * @return 3x1 vector in so(3).
   */
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

  /**
   * @brief Computes the logarithmic map for the current SO3 element.
   * @return 3x1 vector in so(3).
   */
  SO3 inverse() const
  {
    SO3 ret;
    ret._rot = _rot.transpose();
    return ret;
  }

  /**
   * @brief Creates an SO3 element from roll, pitch, and yaw angles.
   * @param rpy Roll, pitch, and yaw angles in radians.
   * @return Corresponding SO3 element.
   */
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

  /**
   * @brief Converts the current SO3 element to roll, pitch, and yaw angles.
   * @return Roll, pitch, and yaw angles in radians.
   */
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

 /**
   * @brief Creates an SO3 element from a quaternion.
   * @param q Quaternion in the order [w, x, y, z].
   * @return Corresponding SO3 element.
   */
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

  /**
   * @brief Converts the current SO3 element to a quaternion.
   * @return Quaternion in the order [w, x, y, z].
   */
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

  /**
   * @brief Computes the left Jacobian matrix for a given 3x1 vector in so(3).
   * @param omega 3x1 vector in so(3).
   * @return 3x3 left Jacobian matrix.
   */
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
          (1.0 - s / theta) * (axis * axis.transpose()) + ((1.0 - c) / theta) * SO3::wedge(axis);
    }

    return J;
  }

  /**
   * @brief Computes the inverse left Jacobian matrix for a given 3x1 vector in so(3).
   * @param omega 3x1 vector in so(3).
   * @return 3x3 inverse left Jacobian matrix.
   */
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

  /**
   * @brief Returns the rotation matrix of the current SO3 element.
   * @return 3x3 rotation matrix.
   */
  Eigen::Matrix3d matrix() const { return _rot; }
  /**
   * @brief Sets the rotation matrix of the SO3 element.
   * @param rot Rotation matrix.
   */
  void from_matrix(const Eigen::Matrix3d& rot) { _rot = rot; }

  // operators

  /**
   * @brief Assignment operator.
   * @param rhs SO3 element to assign.
   * @return Reference to the assigned SO3 element.
   */
  SO3 operator=(const SO3& rhs)
  {
    _rot = rhs._rot;
    return *this;
  }

  /**
   * @brief Compound assignment operator for multiplying with another SO3 element.
   * @param rhs SO3 element to multiply.
   * @return Reference to the modified SO3 element.
   */
  SO3 operator*=(const SO3& rhs)
  {
    _rot *= rhs._rot;
    return *this;
  }

  /**
   * @brief Multiplication operator for two SO3 elements.
   * @param lhs Left-hand side SO3 element.
   * @param rhs Right-hand side SO3 element.
   * @return Result of the multiplication.
   */
  friend SO3 operator*(SO3 lhs, const SO3& rhs) { return lhs *= rhs; }

  /**
   * @brief Multiplication operator for two SO3 elements.
   * @param lhs Left-hand side SO3 element.
   * @param rhs Right-hand side SO3 element.
   * @return Result of the multiplication.
   */
  friend Eigen::Vector3d operator*(Eigen::Vector3d lhs, const SO3& rhs) { return rhs._rot * lhs; }

private:
  Eigen::Matrix3d _rot;  ///< Rotation matrix.
};

}  // namespace nanolie

#endif  // SO3_H
