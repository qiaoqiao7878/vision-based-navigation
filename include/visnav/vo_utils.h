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

#include <visnav/common_types.h>

#include <visnav/calibration.h>

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/triangulation/methods.hpp>

namespace visnav {

void project_landmarks(
    const Sophus::SE3d& current_pose,
    const std::shared_ptr<AbstractCamera<double>>& cam,
    const Landmarks& landmarks, const double cam_z_threshold,
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>&
        projected_points,
    std::vector<TrackId>& projected_track_ids) {
  projected_points.clear();
  projected_track_ids.clear();

  // TODO SHEET 5: project landmarks to the image plane using the current
  // locations of the cameras. Put 2d coordinates of the projected points into
  // projected_points and the corresponding id of the landmark into
  // projected_track_ids.

  for (const auto& lm : landmarks) {
    Eigen::Vector3d lm_cameraframe;
    // transform landmarks from world frame to camera frame
    lm_cameraframe = current_pose.inverse() * lm.second.p;
    // Ignore all points that are behind the camera
    if (lm_cameraframe.z() >= cam_z_threshold) {
      Eigen::Vector2d p_2d;
      p_2d = cam->project(lm_cameraframe);
      // ignore all points that project outside the image
      if ((0 <= p_2d.x()) && (p_2d.x() <= cam->width()) && (p_2d.y() >= 0) &&
          (p_2d.y() <= cam->height())) {
        projected_track_ids.push_back(lm.first);
        projected_points.push_back(p_2d);
      }
    }
  }
  UNUSED(current_pose);
  UNUSED(cam);
  UNUSED(landmarks);
  UNUSED(cam_z_threshold);
}

void find_matches_landmarks(
    const KeypointsData& kdl, const Landmarks& landmarks,
    const Corners& feature_corners,
    const std::vector<Eigen::Vector2d,
                      Eigen::aligned_allocator<Eigen::Vector2d>>&
        projected_points,
    const std::vector<TrackId>& projected_track_ids,
    const double match_max_dist_2d, const int feature_match_max_dist,
    const double feature_match_test_next_best, LandmarkMatchData& md) {
  md.matches.clear();

  // TODO SHEET 5: Find the matches between projected landmarks and detected
  // keypoints in the current frame. For every detected keypoint search for
  // matches inside a circle with radius match_max_dist_2d around the point
  // location. For every landmark the distance is the minimal distance between
  // the descriptor of the current point and descriptors of all observations of
  // the landmarks. The feature_match_max_dist and feature_match_test_next_best
  // should be used to filter outliers the same way as in exercise 3. You should
  // fill md.matches with <featureId,trackId> pairs for the successful matches
  // that pass all tests.

  std::vector<int> min_dis(kdl.corners.size(), 256);
  std::vector<int> second_min_dis(kdl.corners.size(), 256);
  std::vector<TrackId> min_distance_trackId(kdl.corners.size());

  // for every detected point
  for (size_t i = 0; i < kdl.corners.size(); i++) {
    Eigen::Vector2d keypoint;
    keypoint = kdl.corners[i];
    std::bitset<256> keypoint_descriptor;
    keypoint_descriptor = kdl.corner_descriptors[i];
    // for every projected landmarks
    for (size_t j = 0; j < projected_points.size(); j++) {
      TrackId tid = projected_track_ids[j];
      Landmark lm = landmarks.find(tid)->second;
      Eigen::Vector2d projectpoint;
      projectpoint = projected_points[j];

      // if the projection of landmark is inside the circle
      if ((keypoint - projectpoint).norm() <= match_max_dist_2d) {
        // store the minimal distance between the descriptor of the current
        // point and descriptors of all observations of the checked landmark.
        int min_dis_lm = 256;
        // for all observations of the landmark
        for (auto& obs : lm.obs) {
          TimeCamId tcid = obs.first;
          FeatureId fid_obs = obs.second;

          std::bitset<256> obs_descriptor =
              feature_corners.find(tcid)->second.corner_descriptors[fid_obs];
          std::bitset<256> t = obs_descriptor ^ keypoint_descriptor;
          int distance = t.count();

          if (distance < min_dis_lm) {
            min_dis_lm = distance;
          }
        }

        if (min_dis_lm < second_min_dis[i]) {
          second_min_dis[i] = min_dis_lm;
        }
        if (min_dis_lm < min_dis[i]) {
          second_min_dis[i] = min_dis[i];
          min_dis[i] = min_dis_lm;
          min_distance_trackId[i] = tid;
        }
      }
    }
  }
  for (size_t i = 0; i < kdl.corners.size(); i++) {
    if (min_dis[i] < feature_match_max_dist) {
      if (second_min_dis[i] >= min_dis[i] * feature_match_test_next_best) {
        TrackId tid = min_distance_trackId[i];
        FeatureId fid = i;
        md.matches.push_back(std::make_pair(fid, tid));
      }
    }
  }
  UNUSED(kdl);
  UNUSED(landmarks);
  UNUSED(feature_corners);
  UNUSED(projected_points);
  UNUSED(projected_track_ids);
  UNUSED(match_max_dist_2d);
  UNUSED(feature_match_max_dist);
  UNUSED(feature_match_test_next_best);
}

void localize_camera(const std::shared_ptr<AbstractCamera<double>>& cam,
                     const KeypointsData& kdl, const Landmarks& landmarks,
                     const double reprojection_error_pnp_inlier_threshold_pixel,
                     LandmarkMatchData& md) {
  md.inliers.clear();

  if (md.matches.size() == 0) {
    md.T_w_c = Sophus::SE3d();
    return;
  }

  // TODO SHEET 5: Find the pose (md.T_w_c) and the inliers (md.inliers) using
  // the landmark to keypoints matches and PnP. This should be similar to the
  // localize_camera in exercise 4 but in this execise we don't explicitelly
  // have tracks.

  opengv::bearingVectors_t bearingVectors;
  opengv::points_t points;
  // for every track id
  std::vector<TrackId> tids;
  std::vector<FeatureId> fids;

  for (const auto& mt : md.matches) {
    // find the landmark with this track id
    TrackId tid = mt.second;
    tids.push_back(tid);
    points.push_back(landmarks.find(tid)->second.p);
    FeatureId fid = mt.first;
    fids.push_back(fid);
    Eigen::Vector2d p_2d = kdl.corners[fid];
    Eigen::Vector3d p_3d = cam->unproject(p_2d);
    bearingVectors.push_back(p_3d);
  }

  // create the central adapter

  opengv::absolute_pose::CentralAbsoluteAdapter adapter(bearingVectors, points);
  // create a Ransac object
  opengv::sac::Ransac<
      opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
      ransac;
  // create an AbsolutePoseSacProblem
  // (algorithm is selectable: KNEIP, GAO, or EPNP)
  std::shared_ptr<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
      absposeproblem_ptr(
          new opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem(
              adapter, opengv::sac_problems::absolute_pose::
                           AbsolutePoseSacProblem::KNEIP));
  // run ransac
  ransac.sac_model_ = absposeproblem_ptr;
  double focal_length = 500;
  ransac.threshold_ =
      1.0 -
      cos(atan(reprojection_error_pnp_inlier_threshold_pixel / focal_length));
  ransac.computeModel();

  // non-linear optimization (using all inliers)

  adapter.sett(ransac.model_coefficients_.matrix().col(3));
  adapter.setR(ransac.model_coefficients_.matrix().block(0, 0, 3, 3));

  opengv::transformation_t nonlinear_transformation =
      opengv::absolute_pose::optimize_nonlinear(adapter, ransac.inliers_);
  // reselect inliers
  ransac.sac_model_->selectWithinDistance(nonlinear_transformation,
                                          ransac.threshold_, ransac.inliers_);

  std::vector<int> inlier_index;
  inlier_index = ransac.inliers_;

  for (size_t i = 0; i < inlier_index.size(); i++) {
    md.inliers.push_back(
        std::make_pair(fids[inlier_index[i]], tids[inlier_index[i]]));
  }

  // store translation
  md.T_w_c.translation() = nonlinear_transformation.matrix().col(3);

  // store rotation matrix
  Eigen::Matrix3d rotation_matrix;
  rotation_matrix = nonlinear_transformation.matrix().block(0, 0, 3, 3);
  md.T_w_c.so3() = rotation_matrix;

  UNUSED(cam);
  UNUSED(kdl);
  UNUSED(landmarks);
  UNUSED(reprojection_error_pnp_inlier_threshold_pixel);
}

void add_new_landmarks(const TimeCamId tcidl, const TimeCamId tcidr,
                       const KeypointsData& kdl, const KeypointsData& kdr,
                       const Calibration& calib_cam, const MatchData& md_stereo,
                       const LandmarkMatchData& md, Landmarks& landmarks,
                       TrackId& next_landmark_id) {
  // input should be stereo pair
  assert(tcidl.cam_id == 0);
  assert(tcidr.cam_id == 1);

  const Sophus::SE3d T_0_1 = calib_cam.T_i_c[0].inverse() * calib_cam.T_i_c[1];
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

  // TODO SHEET 5: Add new landmarks and observations. Here md_stereo contains
  // stereo matches for the current frame and md contains feature to landmark
  // matches for the left camera (camera 0). For all inlier feature to landmark
  // matches add the observations to the existing landmarks. If the left
  // camera's feature appears also in md_stereo.inliers, then add both
  // observations. For all inlier stereo observations that were not added to the
  // existing landmarks, triangulate and add new landmarks. Here
  // next_landmark_id is a running index of the landmarks, so after adding a new
  // landmark you should always increase next_landmark_id by 1.

  // store the ids that need to be added as new landmark

  std::vector<TrackId> new_track_ids;
  std::vector<FeatureId> new_feature_ids_left;
  std::vector<FeatureId> new_feature_ids_right;

  //  std::vector<std::pair<FeatureId, TrackId>> inliers;
  for (const auto& in : md.inliers) {
    FeatureId fid = in.first;
    TrackId tid = in.second;
    // if landmark exist
    if (landmarks.count(tid) != 0) {
      // add left camera to the landmark
      landmarks.find(tid)->second.obs.insert(std::make_pair(tcidl, fid));
      for (const auto& stereo_in : md_stereo.inliers) {
        // If the left camera's feature appears also in md_stereo.inliers
        if (fid == stereo_in.first) {
          // add right camera to the landmark
          landmarks.find(tid)->second.obs.insert(
              std::make_pair(tcidr, stereo_in.second));
        }
      }
    }
  }
  // for all stereo pairs
  for (const auto& stereo_in : md_stereo.inliers) {
    new_track_ids.push_back(next_landmark_id);
    new_feature_ids_left.push_back(stereo_in.first);
    new_feature_ids_right.push_back(stereo_in.second);
    next_landmark_id++;
    for (const auto& in : md.inliers) {
      // if this stereo not in existing landmark
      if (stereo_in.first == in.first) {
        new_track_ids.pop_back();
        new_feature_ids_left.pop_back();
        new_feature_ids_right.pop_back();
        next_landmark_id--;
      }
    }
  }
  opengv::bearingVectors_t bl;
  opengv::bearingVectors_t br;
  for (size_t i = 0; i < new_feature_ids_left.size(); i++) {
    FeatureId fidl = new_feature_ids_left[i];
    FeatureId fidr = new_feature_ids_right[i];
    Eigen::Vector2d p_2d_l = kdl.corners[fidl];
    Eigen::Vector2d p_2d_r = kdr.corners[fidr];
    Eigen::Vector3d p_3d_l =
        calib_cam.intrinsics[tcidl.cam_id]->unproject(p_2d_l);
    Eigen::Vector3d p_3d_r =
        calib_cam.intrinsics[tcidr.cam_id]->unproject(p_2d_r);
    bl.push_back(p_3d_l);
    br.push_back(p_3d_r);
  }
  opengv::relative_pose::CentralRelativeAdapter adapter(bl, br);
  adapter.setR12(R_0_1);
  adapter.sett12(t_0_1);

  // point_t opengv::triangulation::triangulate(const
  // relative_pose::RelativeAdapterBase & adapter,size_t index)
  // Compute the position of a 3D point seen from two viewpoints. Linear
  // Method.

  // for every new_track_ids
  for (size_t i = 0; i < new_track_ids.size(); i++) {
    // get The landmark (3D point) expressed in the first viewpoint.
    Eigen::Vector3d point = opengv::triangulation::triangulate(adapter, i);

    // using Landmarks = std::unordered_map<TrackId, Landmark>;
    Landmark lm;
    // store landmark position in camera frame
    lm.p = point;

    // using FeatureTrack = std::map<TimeCamId, FeatureId>;
    lm.obs.insert(std::make_pair(tcidl, new_feature_ids_left[i]));
    lm.obs.insert(std::make_pair(tcidr, new_feature_ids_right[i]));
    landmarks.insert(std::make_pair(new_track_ids[i], lm));
  }

  UNUSED(tcidl);
  UNUSED(tcidr);
  UNUSED(kdl);
  UNUSED(kdr);
  UNUSED(calib_cam);
  UNUSED(md_stereo);
  UNUSED(md);
  UNUSED(landmarks);
  UNUSED(next_landmark_id);
  UNUSED(t_0_1);
  UNUSED(R_0_1);
}

void remove_old_keyframes(const TimeCamId tcidl, const int max_num_kfs,
                          Cameras& cameras, Landmarks& landmarks,
                          Landmarks& old_landmarks,
                          std::set<FrameId>& kf_frames) {
  kf_frames.emplace(tcidl.t_ns);

  // TODO SHEET 5: Remove old cameras and observations if the number of
  // keyframe pairs (left and right image is a pair) is larger than
  // max_num_kfs. The ids of all the keyframes that are currently in the
  // optimization should be stored in kf_frames. Removed keyframes should be
  // removed from cameras and landmarks with no left observations should be
  // moved to old_landmarks.

  // using FrameId = int64_t;

  int size_kf = kf_frames.size();
  int num_delete_kf = 0;
  for (auto& kf : kf_frames) {
    num_delete_kf++;
    if (num_delete_kf > size_kf - max_num_kfs) {
      break;
    }
    TimeCamId tcid_delete_l;
    tcid_delete_l.cam_id = 0;
    tcid_delete_l.t_ns = kf;
    cameras.erase(tcid_delete_l);
    TimeCamId tcid_delete_r;
    tcid_delete_r.cam_id = 1;
    tcid_delete_r.t_ns = kf;
    cameras.erase(tcid_delete_r);
    kf_frames.erase(kf);
    for (auto& lm : landmarks) {
      lm.second.obs.erase(tcid_delete_l);
      lm.second.obs.erase(tcid_delete_r);
    }
  }

  for (auto& lm : landmarks) {
    if (lm.second.obs.size() == 0) {
      old_landmarks.insert(lm);
      landmarks.erase(lm.first);
    }
  }
  UNUSED(max_num_kfs);
  UNUSED(cameras);
  UNUSED(landmarks);
  UNUSED(old_landmarks);
}  // namespace visnav
}  // namespace visnav
