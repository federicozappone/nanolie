#ifndef SE3_H
#define SE3_H

#include <Eigen/Core>
#include "Eigen/src/Core/Matrix.h"
#include "so3.h"

typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;

namespace nanolie {

class SE3
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

public:
  SE3() : _trans(Eigen::Vector3d::Zero()) {}
  SE3(const SE3& other) : _rot(other._rot), _trans(other._trans) {}
  SE3(const Eigen::Matrix3d& rot, const Eigen::Vector3d& trans) : _rot(rot), _trans(trans) {}
  SE3(const Eigen::Matrix4d& mat) : _rot(mat.block<3, 3>(0, 0)), _trans(mat.block<3, 1>(0, 3)) {}

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

  static Vector6d vee(const Eigen::Matrix4d& T)
  {
    Vector6d xi;
    xi << T(2, 1), T(0, 2), T(1, 0), T(0, 3), T(1, 3), T(2, 3);
    return xi;
  }

  static SE3 exp(const Vector6d& se3)
  {
    SE3 ret;
    Eigen::Vector3d omega(se3[3], se3[4], se3[5]);
    Eigen::Vector3d v(se3[0], se3[1], se3[2]);

    ret._rot = SO3::exp(omega);
    ret._trans = SO3::left_jacobian(omega) * v;

    return ret;
  }

  Vector6d log() const
  {
    Eigen::Vector3d p = _trans;
    Eigen::Vector3d omega = _rot.log();

    Eigen::Vector3d v = SO3::inverse_left_jacobian(omega) * p;

    Vector6d se3;
    se3 << v[0], v[1], v[2], omega[0], omega[1], omega[2];

    return se3;
  }

  SE3 inverse()
  {
    SE3 T_inv;
    T_inv._rot = _rot.inverse();
    T_inv._trans = -_rot.inverse().matrix() * _trans;
    return T_inv;
  }

  Matrix6d adjoint() const
  {
    Matrix6d adj;
    adj.block<3, 3>(0, 0) = _rot.matrix();
    adj.block<3, 3>(3, 3) = _rot.matrix();
    adj.block<3, 3>(3, 0) = Eigen::Matrix3d::Zero();
    adj.block<3, 3>(0, 3) = _rot.matrix() * SO3::wedge(_trans);
    return adj;
  }

  Matrix6d dual_adjoint() const
  {
    Matrix6d dual_adj;
    dual_adj.block<3, 3>(0, 0) = _rot.matrix().transpose();
    dual_adj.block<3, 3>(3, 3) = _rot.matrix().transpose();
    dual_adj.block<3, 3>(3, 0) = Eigen::Matrix3d::Zero();
    dual_adj.block<3, 3>(0, 3) = SO3::wedge(_trans) * _rot.matrix().transpose();
    return dual_adj;
  }

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

  Eigen::Matrix4d matrix() const
  {
    Eigen::Matrix4d ret = Eigen::Matrix4d::Zero();
    ret.block<3, 3>(0, 0) = _rot.matrix();
    ret.block<3, 1>(0, 3) = _trans;
    ret(3, 3) = 1.0;
    return ret;
  }

  static SE3 from_matrix(const Eigen::Matrix4d& rhs)
  {
    SE3 ret;
    ret._rot.from_matrix(rhs.block<3, 3>(0, 0));
    ret._trans = rhs.block<3, 1>(0, 3);
    return ret;
  }

  SO3 get_rotation() const { return _rot; }
  Eigen::Vector3d get_translation() const { return _trans; }

  void set_rotation(const SO3& rot) { _rot = rot; }
  void set_rotation(const Eigen::Matrix3d& rot) { _rot.from_matrix(rot); }
  void set_translation(const Eigen::Vector3d& trans) { _trans = trans; }

  void perturbate(const Vector6d& se3)
  {
    SE3 perturbed = SE3::exp(se3) * (*this);
    _rot = perturbed._rot;
    _trans = perturbed._trans;
  }

  // operators

  SE3 operator=(const SE3& rhs)
  {
    this->_rot = rhs._rot;
    this->_trans = rhs._trans;
    return *this;
  }

  SE3 operator*=(const SE3& rhs)
  {
    *this = SE3::from_matrix(this->matrix() * rhs.matrix());
    return *this;
  }

  friend SE3 operator*(SE3 lhs, const SE3& rhs) { return lhs *= rhs; }

  friend Eigen::Vector4d operator*(const SE3& lhs, const Eigen::Vector4d& rhs)
  {
    return lhs.matrix() * rhs;
  }

  friend Eigen::Vector3d operator*(const SE3& lhs, const Eigen::Vector3d& rhs)
  {
    Eigen::Vector4d rhs_homo = Eigen::Vector4d::Ones();
    rhs_homo.head(3) = rhs;
    Eigen::Vector4d res = lhs.matrix() * rhs_homo;
    return res.head(3) / res[3];
  }

private:
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

  SO3 _rot;
  Eigen::Vector3d _trans;
};

}  // namespace nanolie

#endif  // SE3_H
