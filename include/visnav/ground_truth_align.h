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

#include <set>

#include <glog/logging.h>
#include <visnav/common_types.h>

#include <visnav/calibration.h>

namespace visnav {

// Some type definitions:
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix<double, 3, 3>;
using Mat3X = Eigen::Matrix<double, 3, Eigen::Dynamic>;
using ArrX = Eigen::ArrayXd;

/// Compute Sim(3) transformation that aligns the 3D points in model to 3D
/// points in data in the least squares sense using Horn's algorithm. I.e. the
/// Sim(3) transformation T is computed that minimizes the sum over all i of
/// ||T*m_i - d_i||^2, m_i are the column's of model and d_i are the column's of
/// data. Both data and model need to be of the same dimension and have at least
/// 3 columns.
///
/// Optionally computes the translational rmse.
///
/// Note that for the orientation we don't actually use Horn's algorithm, but
/// the one published by Arun
/// (http://post.queensu.ca/~sdb2/PAPERS/PAMI-3DLS-1987.pdf) based on SVD with
/// later extension to cover degenerate cases.
///
/// See the illustrative comparison by Eggert:
/// http://graphics.stanford.edu/~smr/ICP/comparison/eggert_comparison_mva97.pdf
Sophus::SE3d align_points_se3(const Eigen::Ref<const Mat3X> &data,
                              const Eigen::Ref<const Mat3X> &model,
                              ErrorMetricValue *ate) {
  CHECK_EQ(data.cols(), model.cols());
  CHECK_GE(data.cols(), 3);

  // 0. Centroids
  const Vec3 centroid_data = data.rowwise().mean();
  const Vec3 centroid_model = model.rowwise().mean();

  // center both clouds to 0 centroid
  const Mat3X data_centered = data.colwise() - centroid_data;
  const Mat3X model_centered = model.colwise() - centroid_model;

  // 1. Rotation

  // sum of outer products of columns
  const Mat3 W = data_centered * model_centered.transpose();

  const auto svd = W.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);

  // last entry to ensure we don't get a reflection, only rotations
  const Mat3 S = Eigen::DiagonalMatrix<double, 3, 3>(
      1, 1,
      svd.matrixU().determinant() * svd.matrixV().determinant() < 0 ? -1 : 1);

  const Mat3 R = svd.matrixU() * S * svd.matrixV().transpose();

  const Mat3X model_rotated = R * model_centered;

  // 3. Translation
  const Vec3 t = centroid_data - R * centroid_model;

  // 4. Translational error
  if (ate) {
    // static_assert(ArrX::ColsAtCompileTime == 1);

    const Mat3X diff = data - ((R * model).colwise() + t);
    const ArrX errors = diff.colwise().norm().transpose();
    auto &ref = *ate;
    ref.rmse = std::sqrt(errors.square().sum() / errors.rows());
    ref.mean = errors.mean();
    ref.min = errors.minCoeff();
    ref.max = errors.maxCoeff();
    ref.count = errors.rows();
  }

  return Sophus::SE3d(R, t);
}  // namespace visnav

// Interface to construct the matrices of camera centers from a list of poses /
// cameras using calibration.
Sophus::SE3d align_cameras_se3(const Cameras &cameras_ground_truth,
                               const Cameras &cameras, ErrorMetricValue *ate) {
  const Eigen::Index num_cameras = cameras.size() / 2;

  Mat3X reference_centers(3, num_cameras);
  Mat3X camera_centers(3, num_cameras);

  size_t i = 0;
  for (const auto &cam : cameras) {
    if (cam.first.cam_id == 0) {
      const auto &T_w_c = cameras_ground_truth.at(cam.first).T_w_c;
      reference_centers.col(i) = T_w_c.translation();
      camera_centers.col(i) = cam.second.T_w_c.translation();
      i++;
    }
  }

  return align_points_se3(reference_centers, camera_centers, ate);
}
}  // namespace visnav
