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

#include <bitset>
#include <set>

#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>

#include <visnav/camera_models.h>
#include <visnav/common_types.h>

namespace visnav {

void computeEssential(const Sophus::SE3d& T_0_1, Eigen::Matrix3d& E) {
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

  // TODO SHEET 3: compute essential matrix
  const Eigen::Vector3d t_0_1_n = t_0_1.normalized();
  Eigen::Matrix3d t_0_1_hat;
  t_0_1_hat << 0, -t_0_1_n[2], t_0_1_n[1], t_0_1_n[2], 0, -t_0_1_n[0],
      -t_0_1_n[1], t_0_1_n[0], 0;
  E = t_0_1_hat * R_0_1;
  UNUSED(E);
  UNUSED(t_0_1);
  UNUSED(R_0_1);
}

void findInliersEssential(const KeypointsData& kd1, const KeypointsData& kd2,
                          const std::shared_ptr<AbstractCamera<double>>& cam1,
                          const std::shared_ptr<AbstractCamera<double>>& cam2,
                          const Eigen::Matrix3d& E,
                          double epipolar_error_threshold, MatchData& md) {
  md.inliers.clear();

  for (size_t j = 0; j < md.matches.size(); j++) {
    const Eigen::Vector2d p0_2d = kd1.corners[md.matches[j].first];
    const Eigen::Vector2d p1_2d = kd2.corners[md.matches[j].second];

    // TODO SHEET 3: determine inliers and store in md.inliers
    Eigen::Vector3d p0_3d = cam1->unproject(p0_2d);
    Eigen::Vector3d p1_3d = cam2->unproject(p1_2d);

    if (abs(p0_3d.transpose() * E * p1_3d) < epipolar_error_threshold) {
      md.inliers.push_back(md.matches[j]);
    }
    UNUSED(cam1);
    UNUSED(cam2);
    UNUSED(E);
    UNUSED(epipolar_error_threshold);
    UNUSED(p0_2d);
    UNUSED(p1_2d);
  }
}

void findInliersRansac(const KeypointsData& kd1, const KeypointsData& kd2,
                       const std::shared_ptr<AbstractCamera<double>>& cam1,
                       const std::shared_ptr<AbstractCamera<double>>& cam2,
                       const double ransac_thresh, const int ransac_min_inliers,
                       MatchData& md) {
  md.inliers.clear();

  // TODO SHEET 3: Run RANSAC with using opengv's CentralRelativePose and store
  // the final inlier indices in md.inliers and the final relative pose in
  // md.T_i_j (normalize translation). If the number of inliers is smaller than
  // ransac_min_inliers, leave md.inliers empty. Note that if the initial RANSAC
  // was successful, you should do non-linear refinement of the model parameters
  // using all inliers, and then re-estimate the inlier set with the refined
  // model parameters.
  opengv::bearingVectors_t p0_3d(md.matches.size());
  opengv::bearingVectors_t p1_3d(md.matches.size());
  Eigen::Vector2d p0_2d;
  Eigen::Vector2d p1_2d;
  for (size_t j = 0; j < md.matches.size(); j++) {
    p0_2d = kd1.corners[md.matches[j].first];
    p1_2d = kd2.corners[md.matches[j].second];
    p0_3d[j] = cam1->unproject(p0_2d);
    p1_3d[j] = cam2->unproject(p1_2d);
  }
  // create the central relative adapter
  opengv::relative_pose::CentralRelativeAdapter adapter(p0_3d, p1_3d);
  // create a RANSAC object
  opengv::sac::Ransac<
      opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem>
      ransac;
  // create a CentralRelativePoseSacProblem
  // (set algorithm to STEWENIUS, NISTER, SEVENPT, or EIGHTPT)
  std::shared_ptr<
      opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem>
      relposeproblem_ptr(
          new opengv::sac_problems::relative_pose::
              CentralRelativePoseSacProblem(
                  adapter, opengv::sac_problems::relative_pose::
                               CentralRelativePoseSacProblem::NISTER));
  // run ransac
  ransac.sac_model_ = relposeproblem_ptr;
  ransac.threshold_ = ransac_thresh;
  ransac.computeModel();

  // non-linear optimization (using all inliers)
  opengv::transformation_t nonlinear_transformation =
      opengv::relative_pose::optimize_nonlinear(adapter, ransac.inliers_);
  // reselect inliers
  ransac.sac_model_->selectWithinDistance(nonlinear_transformation,
                                          ransac.threshold_, ransac.inliers_);
  std::vector<int> inlier_index;
  inlier_index = ransac.inliers_;

  // store inliers
  if (int(inlier_index.size()) < ransac_min_inliers) {
    md.inliers.clear();
  } else {
    for (size_t i = 0; i < inlier_index.size(); i++) {
      md.inliers.push_back(md.matches[inlier_index[i]]);
    }
  }
  // store translation
  md.T_i_j.translation() = nonlinear_transformation.matrix().col(3);
  md.T_i_j.translation() = md.T_i_j.translation().normalized();
  // store rotation matrix
  Eigen::Matrix3d rotation_matrix;
  rotation_matrix = nonlinear_transformation.matrix().block(0, 0, 3, 3);
  md.T_i_j.so3() = rotation_matrix;
  UNUSED(kd1);
  UNUSED(kd2);
  UNUSED(cam1);
  UNUSED(cam2);
  UNUSED(ransac_thresh);
  UNUSED(ransac_min_inliers);
}
}  // namespace visnav
