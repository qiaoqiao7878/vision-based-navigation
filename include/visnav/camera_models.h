/**
BSD 3-Clause License

Copyright (c) 2018, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <memory>

#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include <visnav/common_types.h>

namespace visnav {

template <typename Scalar>
class AbstractCamera;

template <typename Scalar>
class PinholeCamera : public AbstractCamera<Scalar> {
 public:
  static constexpr size_t N = 8;

  typedef Eigen::Matrix<Scalar, 2, 1> Vec2;
  typedef Eigen::Matrix<Scalar, 3, 1> Vec3;

  typedef Eigen::Matrix<Scalar, N, 1> VecN;

  PinholeCamera() { param.setZero(); }

  PinholeCamera(const VecN& p) { param = p; }

  static PinholeCamera<Scalar> getTestProjections() {
    VecN vec1;
    vec1 << 0.5 * 805, 0.5 * 800, 505, 509, 0, 0, 0, 0;
    PinholeCamera<Scalar> res(vec1);

    return res;
  }

  Scalar* data() { return param.data(); }

  const Scalar* data() const { return param.data(); }

  static std::string getName() { return "pinhole"; }
  std::string name() const { return getName(); }

  virtual Vec2 project(const Vec3& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];

    const Scalar& x = p[0];
    const Scalar& y = p[1];
    const Scalar& z = p[2];

    Vec2 res;

    // TODO SHEET 2: implement camera model
    res[0] = fx * x / z + cx;
    res[1] = fy * y / z + cy;
    UNUSED(fx);
    UNUSED(fy);
    UNUSED(cx);
    UNUSED(cy);
    UNUSED(x);
    UNUSED(y);
    UNUSED(z);

    return res;
  }

  virtual Vec3 unproject(const Vec2& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];

    Vec3 res;

    // TODO SHEET 2: implement camera model
    Scalar mx;
    Scalar my;
    mx = (p[0] - cx) / fx;
    my = (p[1] - cy) / fy;
    res << mx, my, Scalar(1);
    res = res / sqrt(mx * mx + my * my + Scalar(1));
    UNUSED(p);
    UNUSED(fx);
    UNUSED(fy);
    UNUSED(cx);
    UNUSED(cy);

    return res;
  }

  const VecN& getParam() const { return param; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  VecN param;
};

template <typename Scalar = double>
class ExtendedUnifiedCamera : public AbstractCamera<Scalar> {
 public:
  // NOTE: For convenience for serialization and handling different camera
  // models in ceres functors, we use the same parameter vector size for all of
  // them, even if that means that for some certain entries are unused /
  // constant 0.
  static constexpr int N = 8;

  typedef Eigen::Matrix<Scalar, 2, 1> Vec2;
  typedef Eigen::Matrix<Scalar, 3, 1> Vec3;
  typedef Eigen::Matrix<Scalar, 4, 1> Vec4;

  typedef Eigen::Matrix<Scalar, N, 1> VecN;

  ExtendedUnifiedCamera() { param.setZero(); }

  ExtendedUnifiedCamera(const VecN& p) { param = p; }

  static ExtendedUnifiedCamera getTestProjections() {
    VecN vec1;
    vec1 << 0.5 * 500, 0.5 * 500, 319.5, 239.5, 0.51231234, 0.9, 0, 0;
    ExtendedUnifiedCamera res(vec1);

    return res;
  }

  Scalar* data() { return param.data(); }
  const Scalar* data() const { return param.data(); }

  static const std::string getName() { return "eucm"; }
  std::string name() const { return getName(); }

  inline Vec2 project(const Vec3& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];
    const Scalar& alpha = param[4];
    const Scalar& beta = param[5];

    const Scalar& x = p[0];
    const Scalar& y = p[1];
    const Scalar& z = p[2];

    Vec2 res;

    // TODO SHEET 2: implement camera model
    Scalar d;
    d = sqrt(beta * (x * x + y * y) + z * z);
    res[0] = fx * x / (alpha * d + (Scalar(1) - alpha) * z) + cx;
    res[1] = fy * y / (alpha * d + (Scalar(1) - alpha) * z) + cy;

    UNUSED(fx);
    UNUSED(fy);
    UNUSED(cx);
    UNUSED(cy);
    UNUSED(alpha);
    UNUSED(beta);
    UNUSED(x);
    UNUSED(y);
    UNUSED(z);

    return res;
  }

  Vec3 unproject(const Vec2& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];
    const Scalar& alpha = param[4];
    const Scalar& beta = param[5];

    Vec3 res;

    // TODO SHEET 2: implement camera model
    Scalar mx = (p[0] - cx) / fx;
    Scalar my = (p[1] - cy) / fy;
    Scalar r_square = mx * mx + my * my;
    Scalar mz = (Scalar(1) - beta * alpha * alpha * r_square) /
                (alpha * sqrt(Scalar(1) - (Scalar(2) * alpha - Scalar(1)) *
                                              beta * r_square) +
                 (Scalar(1) - alpha));
    res << mx, my, mz;
    res = res / sqrt(mx * mx + my * my + mz * mz);
    UNUSED(p);
    UNUSED(fx);
    UNUSED(fy);
    UNUSED(cx);
    UNUSED(cy);
    UNUSED(alpha);
    UNUSED(beta);

    return res;
  }

  const VecN& getParam() const { return param; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  VecN param;
};

template <typename Scalar>
class DoubleSphereCamera : public AbstractCamera<Scalar> {
 public:
  static constexpr size_t N = 8;

  typedef Eigen::Matrix<Scalar, 2, 1> Vec2;
  typedef Eigen::Matrix<Scalar, 3, 1> Vec3;

  typedef Eigen::Matrix<Scalar, N, 1> VecN;

  DoubleSphereCamera() { param.setZero(); }

  DoubleSphereCamera(const VecN& p) { param = p; }

  static DoubleSphereCamera<Scalar> getTestProjections() {
    VecN vec1;
    vec1 << 0.5 * 805, 0.5 * 800, 505, 509, 0.5 * -0.150694, 0.5 * 1.48785, 0,
        0;
    DoubleSphereCamera<Scalar> res(vec1);

    return res;
  }

  Scalar* data() { return param.data(); }
  const Scalar* data() const { return param.data(); }

  static std::string getName() { return "ds"; }
  std::string name() const { return getName(); }

  virtual Vec2 project(const Vec3& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];
    const Scalar& xi = param[4];
    const Scalar& alpha = param[5];

    const Scalar& x = p[0];
    const Scalar& y = p[1];
    const Scalar& z = p[2];

    Vec2 res;

    // TODO SHEET 2: implement camera model
    Scalar d1 = sqrt(x * x + y * y + z * z);
    Scalar d2 = sqrt(x * x + y * y + (xi * d1 + z) * (xi * d1 + z));
    res[0] = fx * x / (alpha * d2 + (Scalar(1) - alpha) * (xi * d1 + z)) + cx;
    res[1] = fy * y / (alpha * d2 + (Scalar(1) - alpha) * (xi * d1 + z)) + cy;

    UNUSED(fx);
    UNUSED(fy);
    UNUSED(cx);
    UNUSED(cy);
    UNUSED(xi);
    UNUSED(alpha);
    UNUSED(x);
    UNUSED(y);
    UNUSED(z);

    return res;
  }

  virtual Vec3 unproject(const Vec2& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];
    const Scalar& xi = param[4];
    const Scalar& alpha = param[5];

    Vec3 res;

    // TODO SHEET 2: implement camera model
    Scalar mx = (p[0] - cx) / fx;
    Scalar my = (p[1] - cy) / fy;
    Scalar r_square = mx * mx + my * my;
    Scalar mz =
        (Scalar(1) - alpha * alpha * r_square) /
        (alpha * sqrt(Scalar(1) - (Scalar(2) * alpha - Scalar(1)) * r_square) +
         Scalar(1) - alpha);
    res << mx, my, mz;
    res = res * ((mz * xi + sqrt(mz * mz + (Scalar(1) - xi * xi) * r_square)) /
                 (mz * mz + r_square));
    res[2] = res[2] - xi;
    UNUSED(p);
    UNUSED(fx);
    UNUSED(fy);
    UNUSED(cx);
    UNUSED(cy);
    UNUSED(xi);
    UNUSED(alpha);
    return res;
  }

  const VecN& getParam() const { return param; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  VecN param;
};

template <typename Scalar = double>
class KannalaBrandt4Camera : public AbstractCamera<Scalar> {
 public:
  static constexpr int N = 8;

  typedef Eigen::Matrix<Scalar, 2, 1> Vec2;
  typedef Eigen::Matrix<Scalar, 3, 1> Vec3;
  typedef Eigen::Matrix<Scalar, 4, 1> Vec4;

  typedef Eigen::Matrix<Scalar, N, 1> VecN;

  KannalaBrandt4Camera() { param.setZero(); }

  KannalaBrandt4Camera(const VecN& p) { param = p; }

  static KannalaBrandt4Camera getTestProjections() {
    VecN vec1;
    vec1 << 379.045, 379.008, 505.512, 509.969, 0.00693023, -0.0013828,
        -0.000272596, -0.000452646;
    KannalaBrandt4Camera res(vec1);

    return res;
  }

  Scalar* data() { return param.data(); }

  const Scalar* data() const { return param.data(); }

  static std::string getName() { return "kb4"; }
  std::string name() const { return getName(); }

  inline Vec2 project(const Vec3& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];
    const Scalar& k1 = param[4];
    const Scalar& k2 = param[5];
    const Scalar& k3 = param[6];
    const Scalar& k4 = param[7];

    const Scalar& x = p[0];
    const Scalar& y = p[1];
    const Scalar& z = p[2];
    Vec2 res;

    // TODO SHEET 2: implement camera model

    Scalar r = sqrt(x * x + y * y);
    Scalar theta = atan2(r, z);

    /*Scalar d_theta = theta + k1 * pow(theta, 3) + k2 * pow(theta, 5) +
                     k3 * pow(theta, 7) + k4 * pow(theta, 9);*/
    Scalar d_theta =
        theta *
        (Scalar(1) +
         theta * theta *
             (k1 + theta * theta *
                       (k2 + theta * theta * (k3 + k4 * theta * theta))));

    res[0] = fx * d_theta * (x / r) + cx;
    res[1] = fy * d_theta * (y / r) + cy;

    UNUSED(fx);
    UNUSED(fy);
    UNUSED(cx);
    UNUSED(cy);
    UNUSED(k1);
    UNUSED(k2);
    UNUSED(k3);
    UNUSED(k4);
    UNUSED(x);
    UNUSED(y);
    UNUSED(z);

    return res;
  }

  Vec3 unproject(const Vec2& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];

    Vec3 res;

    // TODO SHEET 2: implement camera model
    const Scalar& k1 = param[4];
    const Scalar& k2 = param[5];
    const Scalar& k3 = param[6];
    const Scalar& k4 = param[7];
    Scalar mx = (p[0] - cx) / fx;
    Scalar my = (p[1] - cy) / fy;
    Scalar r_u = sqrt(mx * mx + my * my);
    // undoned
    Scalar theta_star;
    Scalar theta = Scalar(0);
    int i = 0;
    while (true) {
      i++;
      theta =
          theta -
          (theta * (Scalar(1) +
                    theta * theta *
                        (k1 + theta * theta *
                                  (k2 + theta * theta *
                                            (k3 + k4 * theta * theta)))) -
           r_u) /
              (Scalar(1) + theta * theta *
                               (Scalar(3) * k1 +
                                theta * theta *
                                    (Scalar(5) * k2 +
                                     theta * theta *
                                         (Scalar(7) * k3 +
                                          Scalar(9) * k4 * theta * theta))));
      if (((theta * (Scalar(1) +
                     theta * theta *
                         (k1 + theta * theta *
                                   (k2 + theta * theta *
                                             (k3 + k4 * theta * theta)))) -
            r_u) < 1e-10) and
          (i > 5)) {
        break;
        if (i > 10) {
          break;
        }
      }
    }

    theta_star = theta;
    res[0] = sin(theta_star) * (mx / r_u);
    res[1] = sin(theta_star) * (my / r_u);
    res[2] = cos(theta_star);

    UNUSED(p);
    UNUSED(fx);
    UNUSED(fy);
    UNUSED(cx);
    UNUSED(cy);

    return res;
  }

  const VecN& getParam() const { return param; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  VecN param;
};

template <typename Scalar>
class AbstractCamera {
 public:
  static constexpr size_t N = 8;

  typedef Eigen::Matrix<Scalar, 2, 1> Vec2;
  typedef Eigen::Matrix<Scalar, 3, 1> Vec3;

  typedef Eigen::Matrix<Scalar, N, 1> VecN;

  virtual ~AbstractCamera() = default;

  virtual Scalar* data() = 0;

  virtual const Scalar* data() const = 0;

  virtual Vec2 project(const Vec3& p) const = 0;

  virtual Vec3 unproject(const Vec2& p) const = 0;

  virtual std::string name() const = 0;

  virtual const VecN& getParam() const = 0;

  inline int width() const { return width_; }
  inline int& width() { return width_; }
  inline int height() const { return height_; }
  inline int& height() { return height_; }

  static std::shared_ptr<AbstractCamera> from_data(const std::string& name,
                                                   const Scalar* sIntr) {
    if (name == DoubleSphereCamera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);
      return std::shared_ptr<AbstractCamera>(
          new DoubleSphereCamera<Scalar>(intr));
    } else if (name == PinholeCamera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);
      return std::shared_ptr<AbstractCamera>(new PinholeCamera<Scalar>(intr));
    } else if (name == KannalaBrandt4Camera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);
      return std::shared_ptr<AbstractCamera>(
          new KannalaBrandt4Camera<Scalar>(intr));
    } else if (name == ExtendedUnifiedCamera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);
      return std::shared_ptr<AbstractCamera>(
          new ExtendedUnifiedCamera<Scalar>(intr));
    } else {
      std::cerr << "Camera model " << name << " is not implemented."
                << std::endl;
      std::abort();
    }
  }

  // Loading from double sphere initialization
  static std::shared_ptr<AbstractCamera> initialize(const std::string& name,
                                                    const Scalar* sIntr) {
    Eigen::Matrix<Scalar, 8, 1> init_intr;

    if (name == DoubleSphereCamera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);

      init_intr = intr;

      return std::shared_ptr<AbstractCamera>(
          new DoubleSphereCamera<Scalar>(init_intr));
    } else if (name == PinholeCamera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);

      init_intr = intr;
      init_intr.template tail<4>().setZero();

      return std::shared_ptr<AbstractCamera>(
          new PinholeCamera<Scalar>(init_intr));
    } else if (name == KannalaBrandt4Camera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);

      init_intr = intr;
      init_intr.template tail<4>().setZero();

      return std::shared_ptr<AbstractCamera>(
          new KannalaBrandt4Camera<Scalar>(init_intr));
    } else if (name == ExtendedUnifiedCamera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);

      init_intr = intr;
      init_intr.template tail<4>().setZero();
      init_intr[4] = 0.5;
      init_intr[5] = 1;

      return std::shared_ptr<AbstractCamera>(
          new ExtendedUnifiedCamera<Scalar>(init_intr));
    } else {
      std::cerr << "Camera model " << name << " is not implemented."
                << std::endl;
      std::abort();
    }
  }

 private:
  // image dimensions
  int width_ = 0;
  int height_ = 0;
};

}  // namespace visnav
