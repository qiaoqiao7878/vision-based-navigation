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

#include <fstream>

#include <tbb/task_scheduler_init.h>

#include <ceres/ceres.h>

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/triangulation/methods.hpp>

#include <visnav/common_types.h>
#include <visnav/serialization.h>

#include <visnav/reprojection.h>
#include <visnav/local_parameterization_se3.hpp>

#include <visnav/tracks.h>

namespace visnav {

// save map with all features and matches
void save_map_file(const std::string& map_path, const Corners& feature_corners,
                   const Matches& feature_matches,
                   const FeatureTracks& feature_tracks,
                   const FeatureTracks& outlier_tracks, const Cameras& cameras,
                   const Landmarks& landmarks) {
  {
    std::ofstream os(map_path, std::ios::binary);

    if (os.is_open()) {
      cereal::BinaryOutputArchive archive(os);
      archive(feature_corners);
      archive(feature_matches);
      archive(feature_tracks);
      archive(outlier_tracks);
      archive(cameras);
      archive(landmarks);

      size_t num_obs = 0;
      for (const auto& kv : landmarks) {
        num_obs += kv.second.obs.size();
      }
      std::cout << "Saved map as " << map_path << " (" << cameras.size()
                << " cameras, " << landmarks.size() << " landmarks, " << num_obs
                << " observations)" << std::endl;
    } else {
      std::cout << "Failed to save map as " << map_path << std::endl;
    }
  }
}

// load map with all features and matches
void load_map_file(const std::string& map_path, Corners& feature_corners,
                   Matches& feature_matches, FeatureTracks& feature_tracks,
                   FeatureTracks& outlier_tracks, Cameras& cameras,
                   Landmarks& landmarks) {
  {
    std::ifstream is(map_path, std::ios::binary);

    if (is.is_open()) {
      cereal::BinaryInputArchive archive(is);
      archive(feature_corners);
      archive(feature_matches);
      archive(feature_tracks);
      archive(outlier_tracks);
      archive(cameras);
      archive(landmarks);

      size_t num_obs = 0;
      for (const auto& kv : landmarks) {
        num_obs += kv.second.obs.size();
      }
      std::cout << "Loaded map from " << map_path << " (" << cameras.size()
                << " cameras, " << landmarks.size() << " landmarks, " << num_obs
                << " observations)" << std::endl;
    } else {
      std::cout << "Failed to load map from " << map_path << std::endl;
    }
  }
}

// Create new landmarks from shared feature tracks if they don't already exist.
// The two cameras must be in the map already.
// Returns the number of newly created landmarks.
int add_new_landmarks_between_cams(const TimeCamId& tcid0,
                                   const TimeCamId& tcid1,
                                   const Calibration& calib_cam,
                                   const Corners& feature_corners,
                                   const FeatureTracks& feature_tracks,
                                   const Cameras& cameras,
                                   Landmarks& landmarks) {
  // shared_track_ids will contain all track ids shared between the two images,
  // including existing landmarks
  std::vector<TrackId> shared_track_ids;

  // find shared feature tracks
  const std::set<TimeCamId> tcids = {tcid0, tcid1};
  if (!GetTracksInImages(tcids, feature_tracks, shared_track_ids)) {
    return 0;
  }

  // at the end of the function this will contain all newly added track ids
  std::vector<TrackId> new_track_ids;

  // TODO SHEET 4: Triangulate all new features and add to the map

  // TrackId: Ids for feature tracks; also used for landmarks created from (some
  // of) the tracks;

  // using TrackId = int64_t;

  // using FeatureTracks = std::unordered_map<TrackId, FeatureTrack>;

  // using FeatureTrack = std::map<TimeCamId, FeatureId>;

  // FeatureId: ids for 2D features detected in images

  // using FeatureId = int;

  // using Corners = tbb::concurrent_unordered_map<TimeCamId,
  // KeypointsData>;

  // using Cameras = std::map<TimeCamId, Camera, std::less<TimeCamId>,
  // Eigen::aligned_allocator<std::pair<const TimeCamId, Camera>>>;

  opengv::bearingVectors_t b0;
  opengv::bearingVectors_t b1;
  for (const auto& tr : shared_track_ids) {
    auto it = landmarks.find(tr);
    // if the track_id doesn't exist in the lankmarks, add it
    if (it == landmarks.end()) {
      new_track_ids.push_back(tr);
    }
  }
  for (const auto& tr : new_track_ids) {
    auto it = feature_tracks.find(tr);

    if (it != feature_tracks.end()) {
      FeatureTrack ft;
      ft = it->second;
      FeatureId fid0 = ft.find(tcid0)->second;
      FeatureId fid1 = ft.find(tcid1)->second;
      Eigen::Vector2d p0_2d = feature_corners.find(tcid0)->second.corners[fid0];
      Eigen::Vector2d p1_2d = feature_corners.find(tcid1)->second.corners[fid1];
      Eigen::Vector3d p0_3d =
          calib_cam.intrinsics[tcid0.cam_id]->unproject(p0_2d);
      Eigen::Vector3d p1_3d =
          calib_cam.intrinsics[tcid1.cam_id]->unproject(p1_2d);
      b0.push_back(p0_3d);
      b1.push_back(p1_3d);
    }
  }

  opengv::relative_pose::CentralRelativeAdapter adapter(b0, b1);
  Sophus::SE3d rel_pose = cameras.find(tcid0)->second.T_w_c.inverse() *
                          cameras.find(tcid1)->second.T_w_c;

  adapter.setR12(rel_pose.so3().matrix());
  adapter.sett12(rel_pose.translation());

  // point_t opengv::triangulation::triangulate(const
  // relative_pose::RelativeAdapterBase & adapter,size_t index)
  // Compute the position of a 3D point seen from two viewpoints. Linear
  // Method.Return The 3D point expressed in the first viewpoint.
  for (size_t i = 0; i < new_track_ids.size(); i++) {
    Eigen::Vector3d point = opengv::triangulation::triangulate(adapter, i);

    // using Landmarks = std::unordered_map<TrackId, Landmark>;
    Landmark lm;
    lm.p = cameras.find(tcid0)->second.T_w_c * point;
    FeatureTrack ft = feature_tracks.find(new_track_ids[i])->second;
    // FeatureTrack obs;
    lm.obs = ft;

    landmarks.insert(std::make_pair(new_track_ids[i], lm));
  }

  UNUSED(calib_cam);
  UNUSED(feature_corners);
  UNUSED(cameras);
  UNUSED(landmarks);

  return new_track_ids.size();
}

// Initialize the scene from a stereo pair, using the known transformation from
// camera calibration. This adds the inital two cameras and triangulates shared
// landmarks.
// Note: in principle we could also initialize a map from another images pair
// using the transformation from the pairwise matching with the 5-point
// algorithm. However, using a stereo pair has the advantage that the map is
// initialized with metric scale.
bool initialize_scene_from_stereo_pair(const TimeCamId& tcid0,
                                       const TimeCamId& tcid1,
                                       const Calibration& calib_cam,
                                       const Corners& feature_corners,
                                       const FeatureTracks& feature_tracks,
                                       Cameras& cameras, Landmarks& landmarks) {
  // check that the two image ids refer to a stereo pair
  if (!(tcid0.t_ns == tcid1.t_ns && tcid0.cam_id != tcid1.cam_id)) {
    std::cerr << "Images " << tcid0 << " and " << tcid1
              << " don't for a stereo pair. Cannot initialize." << std::endl;
    return false;
  }

  // TODO SHEET 4: Initialize scene (add initial cameras and landmarks)
  Camera camera0;
  Sophus::SE3d se3;
  camera0.T_w_c = se3;
  Camera camera1;
  camera1.T_w_c = calib_cam.T_i_c[0].inverse() * calib_cam.T_i_c[1];

  cameras.insert(std::make_pair(tcid0, camera0));
  cameras.insert(std::make_pair(tcid1, camera1));

  UNUSED(calib_cam);
  UNUSED(feature_corners);
  UNUSED(feature_tracks);
  UNUSED(cameras);
  UNUSED(landmarks);

  return true;
}

// Localize a new camera in the map given a set of observed landmarks. We use
// pnp and ransac to localize the camera in the presence of outlier tracks.
// After finding an inlier set with pnp, we do non-linear refinement using all
// inliers and also update the set of inliers using the refined pose.
//
// shared_track_ids already contains those tracks which the new image shares
// with the landmarks (but some might be outliers).
//
// We return the refined pose and the set of track ids for all inliers.
//
// The inlier threshold is given in pixels. See also the opengv documentation on
// how to convert this to a ransac threshold:
// http://laurentkneip.github.io/opengv/page_how_to_use.html#sec_threshold
void localize_camera(
    const TimeCamId& tcid, const std::vector<TrackId>& shared_track_ids,
    const Calibration& calib_cam, const Corners& feature_corners,
    const FeatureTracks& feature_tracks, const Landmarks& landmarks,
    const double reprojection_error_pnp_inlier_threshold_pixel,
    Sophus::SE3d& T_w_c, std::vector<TrackId>& inlier_track_ids) {
  inlier_track_ids.clear();

  // TODO SHEET 4: Localize a new image in a given map
  UNUSED(tcid);
  UNUSED(shared_track_ids);
  UNUSED(calib_cam);
  UNUSED(feature_corners);
  UNUSED(feature_tracks);
  UNUSED(landmarks);
  UNUSED(T_w_c);
  UNUSED(reprojection_error_pnp_inlier_threshold_pixel);
}

struct BundleAdjustmentOptions {
  /// 0: silent, 1: ceres brief report (one line), 2: ceres full report
  int verbosity_level = 1;

  /// update intrinsics or keep fixed
  bool optimize_intrinsics = false;

  /// use huber robust norm or squared norm
  bool use_huber = true;

  /// parameter for huber loss (in pixel)
  double huber_parameter = 1.0;

  /// maximum number of solver iterations
  int max_num_iterations = 20;
};

// Run bundle adjustment to optimize cameras, points, and optionally intrinsics
void bundle_adjustment(const Corners& feature_corners,
                       const BundleAdjustmentOptions& options,
                       const std::set<TimeCamId>& fixed_cameras,
                       Calibration& calib_cam, Cameras& cameras,
                       Landmarks& landmarks) {
  ceres::Problem problem;

  // TODO SHEET 4: Setup optimization problem
  UNUSED(feature_corners);
  UNUSED(options);
  UNUSED(fixed_cameras);
  UNUSED(calib_cam);
  UNUSED(cameras);
  UNUSED(landmarks);

  // Solve
  ceres::Solver::Options ceres_options;
  ceres_options.max_num_iterations = options.max_num_iterations;
  ceres_options.linear_solver_type = ceres::SPARSE_SCHUR;
  ceres_options.num_threads = tbb::task_scheduler_init::default_num_threads();
  ceres::Solver::Summary summary;
  Solve(ceres_options, &problem, &summary);
  switch (options.verbosity_level) {
    // 0: silent
    case 1:
      std::cout << summary.BriefReport() << std::endl;
      break;
    case 2:
      std::cout << summary.FullReport() << std::endl;
      break;
  }
}

}  // namespace visnav
