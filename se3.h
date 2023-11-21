#ifndef SE3_H
#define SE3_H

#include <Eigen/Core>
#include "so3.h"

typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;

namespace nanolie {

/**
 * @brief Represents a 3D rigid body transformation in the Special Euclidean group SE(3).
 */
class SE3
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

public:
  /**
   * @brief Default constructor. Initializes with zero translation.
   */
  SE3() : _trans(Eigen::Vector3d::Zero()) {}
  /**
   * @brief Copy constructor.
   * @param other The SE3 object to copy.
   */
  SE3(const SE3& other) : _rot(other._rot), _trans(other._trans) {}
  /**
   * @brief Constructor with rotation matrix and translation vector.
   * @param rot Rotation matrix.
   * @param trans Translation vector.
   */
  SE3(const Eigen::Matrix3d& rot, const Eigen::Vector3d& trans) : _rot(rot), _trans(trans) {}
  /**
   * @brief Constructor with 4x4 transformation matrix.
   * @param mat 4x4 transformation matrix.
   */
  SE3(const Eigen::Matrix4d& mat) : _rot(mat.block<3, 3>(0, 0)), _trans(mat.block<3, 1>(0, 3)) {}

  /**
   * @brief Computes the wedge operator of a 6x1 vector.
   * @param xi 6x1 vector.
   * @return 4x4 matrix representing the wedge operator.
   */
  static Eigen::Matrix4d wedge(const Vector6d& xi)
  {
    Eigen::Matrix4d Omega;
    // clang-format off
    Omega << 0.0, -xi(2), xi(1), xi(3),
             xi(2), 0.0, -xi(0), xi(4),
            -xi(1), xi(0), 0.0, xi(5),
             0.0, 0.0, 0.0, 0.0;
    // clang-format on
    return Omega;
  }

  /**
   * @brief Computes the vee operator of a 4x4 matrix.
   * @param T 4x4 matrix.
   * @return 6x1 vector representing the vee operator.
   */
  static Vector6d vee(const Eigen::Matrix4d& T)
  {
    Vector6d xi;
    xi << T(2, 1), T(0, 2), T(1, 0), T(0, 3), T(1, 3), T(2, 3);
    return xi;
  }

  /**
   * @brief Computes the exponential map for an element in the Lie algebra se(3).
   * @param se3 6x1 vector in se(3).
   * @return Corresponding SE3 element.
   */
  static SE3 exp(const Vector6d& se3)
  {
    SE3 ret;
    Eigen::Vector3d omega(se3[3], se3[4], se3[5]);
    Eigen::Vector3d v(se3[0], se3[1], se3[2]);

    ret._rot = SO3::exp(omega);
    ret._trans = SO3::left_jacobian(omega) * v;

    return ret;
  }

  /**
   * @brief Computes the inverse of the current SE3 element.
   * @return Inverse of the SE3 element.
   */
  Vector6d log() const
  {
    Eigen::Vector3d p = _trans;
    Eigen::Vector3d omega = _rot.log();

    Eigen::Vector3d v = SO3::inverse_left_jacobian(omega) * p;

    Vector6d se3;
    se3 << v[0], v[1], v[2], omega[0], omega[1], omega[2];

    return se3;
  }

  /**
   * @brief Computes the inverse of the current SE3 element.
   * @return Inverse of the SE3 element.
   */
  SE3 inverse()
  {
    SE3 T_inv;
    T_inv._rot = _rot.inverse();
    T_inv._trans = -_rot.inverse().matrix() * _trans;
    return T_inv;
  }

  /**
   * @brief Computes the adjoint matrix of the current SE3 element.
   * @return 6x6 adjoint matrix.
   */
  Matrix6d adjoint() const
  {
    Matrix6d adj;
    adj.block<3, 3>(0, 0) = _rot.matrix();
    adj.block<3, 3>(3, 3) = _rot.matrix();
    adj.block<3, 3>(3, 0) = Eigen::Matrix3d::Zero();
    adj.block<3, 3>(0, 3) = _rot.matrix() * SO3::wedge(_trans);
    return adj;
  }

  /**
   * @brief Computes the dual adjoint matrix of the current SE3 element.
   * @return 6x6 dual adjoint matrix.
   */
  Matrix6d dual_adjoint() const
  {
    Matrix6d dual_adj;
    dual_adj.block<3, 3>(0, 0) = _rot.matrix().transpose();
    dual_adj.block<3, 3>(3, 3) = _rot.matrix().transpose();
    dual_adj.block<3, 3>(3, 0) = Eigen::Matrix3d::Zero();
    dual_adj.block<3, 3>(0, 3) = SO3::wedge(_trans) * _rot.matrix().transpose();
    return dual_adj;
  }

  /**
   * @brief Computes the left Jacobian matrix for a given 6x1 vector in se(3).
   * @param se3 6x1 vector in se(3).
   * @return 6x6 left Jacobian matrix.
   */
  static Matrix6d left_jacobian(const Vector6d& se3)
  {
    Eigen::Vector3d omega(se3[3], se3[4], se3[5]);
    Eigen::Vector3d v(se3[0], se3[1], se3[2]);

    Eigen::Matrix3d so3_left_jac = SO3::left_jacobian(omega);
    Eigen::Matrix3d Q = SE3::_left_jacobian_Q_matrix(se3);

    Matrix6d J;
    J.block<3, 3>(0, 0) = so3_left_jac;
    J.block<3, 3>(3, 3) = so3_left_jac;
    J.block<3, 3>(0, 3) = Q;
    J.block<3, 3>(3, 0) = Eigen::Matrix3d::Zero();
    return J;
  }

  /**
   * @brief Computes the inverse left Jacobian matrix for a given 6x1 vector in se(3).
   * @param se3 6x1 vector in se(3).
   * @return 6x6 inverse left Jacobian matrix.
   */
  static Matrix6d inverse_left_jacobian(const Vector6d& se3)
  {
    Eigen::Vector3d omega(se3[3], se3[4], se3[5]);
    Eigen::Vector3d v(se3[0], se3[1], se3[2]);

    Eigen::Matrix3d so3_inv_left_jac = SO3::inverse_left_jacobian(omega);
    Eigen::Matrix3d Q = SE3::_left_jacobian_Q_matrix(se3);

    Matrix6d J_inv;
    J_inv.block<3, 3>(0, 0) = so3_inv_left_jac;
    J_inv.block<3, 3>(3, 3) = so3_inv_left_jac;
    J_inv.block<3, 3>(0, 3) = -(so3_inv_left_jac * Q) * so3_inv_left_jac;
    J_inv.block<3, 3>(3, 0) = Eigen::Matrix3d::Zero();
    return J_inv;
  }

  /**
   * @brief Returns the 4x4 homogeneous transformation matrix for the current SE3 element.
   * @return 4x4 homogeneous transformation matrix.
   */
  Eigen::Matrix4d matrix() const
  {
    Eigen::Matrix4d ret = Eigen::Matrix4d::Zero();
    ret.block<3, 3>(0, 0) = _rot.matrix();
    ret.block<3, 1>(0, 3) = _trans;
    ret(3, 3) = 1.0;
    return ret;
  }

  /**
   * @brief Creates an SE3 element from a 4x4 homogeneous transformation matrix.
   * @param rhs 4x4 homogeneous transformation matrix.
   * @return Corresponding SE3 element.
   */
  static SE3 from_matrix(const Eigen::Matrix4d& rhs)
  {
    SE3 ret;
    ret._rot.from_matrix(rhs.block<3, 3>(0, 0));
    ret._trans = rhs.block<3, 1>(0, 3);
    return ret;
  }

  /**
   * @brief Gets the rotation component of the SE3 element.
   * @return SO3 rotation element.
   */
  SO3 get_rotation() const { return _rot; }
  /**
   * @brief Gets the translation component of the SE3 element.
   * @return Translation vector.
   */
  Eigen::Vector3d get_translation() const { return _trans; }

  /**
   * @brief Sets the rotation component of the SE3 element.
   * @param rot SO3 rotation element.
   */
  void set_rotation(const SO3& rot) { _rot = rot; }
  /**
   * @brief Sets the rotation component of the SE3 element using a rotation matrix.
   * @param rot Rotation matrix.
   */
  void set_rotation(const Eigen::Matrix3d& rot) { _rot.from_matrix(rot); }
  /**
   * @brief Sets the translation component of the SE3 element.
   * @param trans Translation vector.
   */
  void set_translation(const Eigen::Vector3d& trans) { _trans = trans; }

  /**
   * @brief Perturbs the current SE3 element by a given 6x1 vector in se(3).
   * @param se3 6x1 vector in se(3).
   */
  void perturbate(const Vector6d& se3)
  {
    SE3 perturbed = SE3::exp(se3) * (*this);
    _rot = perturbed._rot;
    _trans = perturbed._trans;
  }

  // operators
  
  /**
   * @brief Assignment operator.
   * @param rhs SE3 element to assign.
   * @return Reference to the assigned SE3 element.
   */
  SE3 operator=(const SE3& rhs)
  {
    this->_rot = rhs._rot;
    this->_trans = rhs._trans;
    return *this;
  }

  /**
   * @brief Compound assignment operator for multiplying with another SE3 element.
   * @param rhs SE3 element to multiply.
   * @return Reference to the modified SE3 element.
   */
  SE3 operator*=(const SE3& rhs)
  {
    *this = SE3::from_matrix(this->matrix() * rhs.matrix());
    return *this;
  }

  /**
   * @brief Multiplication operator for two SE3 elements.
   * @param lhs Left-hand side SE3 element.
   * @param rhs Right-hand side SE3 element.
   * @return Result of the multiplication.
   */
  friend SE3 operator*(SE3 lhs, const SE3& rhs) { return lhs *= rhs; }

  /**
   * @brief Multiplication operator for an SE3 element and a 4x1 vector.
   * @param lhs SE3 element.
   * @param rhs 4x1 vector.
   * @return Result of the multiplication.
   */
  friend Eigen::Vector4d operator*(const SE3& lhs, const Eigen::Vector4d& rhs)
  {
    return lhs.matrix() * rhs;
  }

  /**
   * @brief Multiplication operator for an SE3 element and a 3x1 vector.
   * @param lhs SE3 element.
   * @param rhs 3x1 vector.
   * @return Result of the multiplication.
   */
  friend Eigen::Vector3d operator*(const SE3& lhs, const Eigen::Vector3d& rhs)
  {
    Eigen::Vector4d rhs_homo = Eigen::Vector4d::Ones();
    rhs_homo.head(3) = rhs;
    Eigen::Vector4d res = lhs.matrix() * rhs_homo;
    return res.head(3) / res[3];
  }

private:
  /**
   * @brief Helper function to compute a part of the left Jacobian Q matrix.
   * @param se3 6x1 vector in se(3).
   * @return 3x3 matrix.
   */
  static Eigen::Matrix3d _left_jacobian_Q_matrix(const Vector6d& se3)
  {
    Eigen::Vector3d rho(se3[3], se3[4], se3[5]);
    Eigen::Vector3d phi(se3[0], se3[1], se3[2]);

    Eigen::Matrix3d rx = SO3::wedge(rho);
    Eigen::Matrix3d px = SO3::wedge(phi);

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

  SO3 _rot;                  ///< Rotation component.
  Eigen::Vector3d _trans;    ///< Translation component.
};

}  // namespace nanolie

#endif  // SE3_H
