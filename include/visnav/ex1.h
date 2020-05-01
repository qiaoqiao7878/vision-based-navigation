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

#include <sophus/se3.hpp>

#include <visnav/common_types.h>

namespace visnav {

// Implement exp for SO(3)
template <class T>
Eigen::Matrix<T, 3, 3> user_implemented_expmap(
    const Eigen::Matrix<T, 3, 1>& xi) {
  // TODO SHEET 1: implement

  Eigen::Matrix<T, 3, 3> w_hat;
  w_hat << 0, -xi(2), xi(1), xi(2), 0, -xi(0), -xi(1), xi(0), 0;

  double theta;
  theta = sqrt(xi(0) * xi(0) + xi(1) * xi(1) + xi(2) * xi(2));
  Eigen::Matrix<T, 3, 3> expmap;
  expmap = Eigen::MatrixXd::Identity(3, 3);
  if (theta > 1e-8) {
    expmap = expmap + sin(theta) / theta * w_hat;
    expmap = expmap + (1 - cos(theta)) / (theta * theta) * w_hat * w_hat;
  }
  UNUSED(xi);
  return expmap;
}

// Implement log for SO(3)
template <class T>
Eigen::Matrix<T, 3, 1> user_implemented_logmap(
    const Eigen::Matrix<T, 3, 3>& mat) {
  // TODO SHEET 1: implement
  Eigen::Matrix<T, 3, 1> w = Eigen::MatrixXd::Zero(3, 1);
  double theta = acos((mat(0, 0) + mat(1, 1) + mat(2, 2) - 1) / 2);
  if (theta > 1e-8) {
    w << mat(2, 1) - mat(1, 2), mat(0, 2) - mat(2, 0), mat(1, 0) - mat(0, 1);
    w = theta * w / (2 * sin(theta));
  }
  UNUSED(mat);
  return w;
}

// Implement exp for SE(3)
template <class T>
Eigen::Matrix<T, 4, 4> user_implemented_expmap(
    const Eigen::Matrix<T, 6, 1>& xi) {
  // TODO SHEET 1: implement
  Eigen::Matrix<T, 4, 4> expmap_SE3 = Eigen::MatrixXd::Identity(4, 4);

  Eigen::Matrix<T, 3, 1> v;
  v << xi(0), xi(1), xi(2);
  Eigen::Matrix<T, 3, 1> w;
  w << xi(3), xi(4), xi(5);

  Eigen::Matrix<T, 3, 3> w_hat;
  w_hat << 0, -w(2), w(1), w(2), 0, -w(0), -w(1), w(0), 0;

  double theta = sqrt(w(0) * w(0) + w(1) * w(1) + w(2) * w(2));

  if (theta > 1e-12) {
    expmap_SE3.block(0, 0, 3, 3) += sin(theta) / theta * w_hat;
    expmap_SE3.block(0, 0, 3, 3) +=
        (1 - cos(theta)) / (theta * theta) * w_hat * w_hat;
  }

  Eigen::Matrix<T, 3, 3> J = Eigen::MatrixXd::Identity(3, 3);

  if (theta > 1e-12) {
    J += (1 - cos(theta)) / (theta * theta) * w_hat;
    J += (theta - sin(theta)) / pow(theta, 3) * w_hat * w_hat;
  }

  expmap_SE3.block(0, 3, 3, 1) = J * v;
  UNUSED(xi);
  return expmap_SE3;
}

// Implement log for SE(3)
template <class T>
Eigen::Matrix<T, 6, 1> user_implemented_logmap(
    const Eigen::Matrix<T, 4, 4>& mat) {
  // TODO SHEET 1: implement

  Eigen::Matrix<T, 3, 1> t;
  t << mat(0, 3), mat(1, 3), mat(2, 3);
  Eigen::Matrix<T, 3, 1> w = Eigen::MatrixXd::Zero(3, 1);

  double theta = acos((mat(0, 0) + mat(1, 1) + mat(2, 2) - 1) / 2);

  if (theta > 1e-20) {
    w << mat(2, 1) - mat(1, 2), mat(0, 2) - mat(2, 0), mat(1, 0) - mat(0, 1);
    w = theta * w / (2 * sin(theta));
  }

  Eigen::Matrix<T, 3, 3> w_hat;
  w_hat << 0, -w(2), w(1), w(2), 0, -w(0), -w(1), w(0), 0;

  Eigen::Matrix<T, 3, 3> J_inv = Eigen::MatrixXd::Identity(3, 3);

  J_inv = J_inv - 0.5 * w_hat;

  if (theta > 1e-20) {
    J_inv = J_inv + ((1 / (theta * theta)) -
                     ((1 + cos(theta)) / (2 * theta * sin(theta)))) *
                        w_hat * w_hat;
  }

  Eigen::Matrix<T, 3, 1> v;
  v = J_inv * t;

  Eigen::Matrix<T, 6, 1> xi;
  xi << v(0), v(1), v(2), w(0), w(1), w(2);
  UNUSED(mat);
  return xi;
}

}  // namespace visnav
