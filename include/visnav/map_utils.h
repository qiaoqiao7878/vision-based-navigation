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

#include <visnav/covisibility_graph.h>

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
  // Method.

  // for every new_track_ids
  for (size_t i = 0; i < new_track_ids.size(); i++) {
    // get The landmark (3D point) expressed in the first viewpoint.
    Eigen::Vector3d point = opengv::triangulation::triangulate(adapter, i);

    // using Landmarks = std::unordered_map<TrackId, Landmark>;
    Landmark lm;
    // store landmark position in world frame
    lm.p = cameras.find(tcid0)->second.T_w_c * point;
    // using FeatureTracks = std::unordered_map<TrackId, FeatureTrack>;

    // using FeatureTrack = std::map<TimeCamId, FeatureId>;
    FeatureTrack featuretrack = feature_tracks.find(new_track_ids[i])->second;
    // FeatureTrack obs;
    for (const auto& ft : featuretrack) {
      // check if the camId is one of the existing cameras in the map
      if (cameras.count(ft.first) != 0) {
        lm.obs.insert(ft);
      }
    }
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
  camera1.T_w_c =
      calib_cam.T_i_c[tcid0.cam_id].inverse() * calib_cam.T_i_c[tcid1.cam_id];

  cameras.insert(std::make_pair(tcid0, camera0));
  cameras.insert(std::make_pair(tcid1, camera1));
  landmarks.clear();

  visnav::add_new_landmarks_between_cams(tcid0, tcid1, calib_cam,
                                         feature_corners, feature_tracks,
                                         cameras, landmarks);
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
  opengv::bearingVectors_t bearingVectors;
  opengv::points_t points;
  // for every track id

  for (const auto& tr : shared_track_ids) {
    // find the landmark with this track id
    points.push_back(landmarks.find(tr)->second.p);
    FeatureId fid = feature_tracks.find(tr)->second.find(tcid)->second;
    Eigen::Vector2d p_2d = feature_corners.find(tcid)->second.corners[fid];
    Eigen::Vector3d p_3d = calib_cam.intrinsics[tcid.cam_id]->unproject(p_2d);
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
    inlier_track_ids.push_back(shared_track_ids[inlier_index[i]]);
  }

  // store translation
  T_w_c.translation() = nonlinear_transformation.matrix().col(3);

  // store rotation matrix
  Eigen::Matrix3d rotation_matrix;
  rotation_matrix = nonlinear_transformation.matrix().block(0, 0, 3, 3);
  T_w_c.so3() = rotation_matrix;
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

// Run bundle adjustment to optimize cameras, points, and optionally
// intrinsics
void bundle_adjustment(const Corners& feature_corners,
                       const BundleAdjustmentOptions& options,
                       const std::set<TimeCamId>& fixed_cameras,
                       Calibration& calib_cam, Cameras& cameras,
                       Landmarks& landmarks) {
  ceres::Problem problem;

  // TODO SHEET 4: Setup optimization problem

  // Adding parameter T_w_c for every camera
  for (auto& ca : cameras) {
    problem.AddParameterBlock(ca.second.T_w_c.data(),
                              ca.second.T_w_c.num_parameters,
                              new Sophus::test::LocalParameterizationSE3);
  }

  // adding intrinsics parameters use maximum number 8
  for (auto& intr : calib_cam.intrinsics) {
    problem.AddParameterBlock(intr->data(), 8);
    // optimizing intrinstics or not depending on options
    if (options.optimize_intrinsics == false) {
      problem.SetParameterBlockConstant(intr->data());
    }
  }
  // if this camrera is in the set of fixed cameras, set parameter to be
  // constant
  for (auto& fc : fixed_cameras) {
    problem.SetParameterBlockConstant(cameras.find(fc)->second.T_w_c.data());
  }
  // get the name of camera model, cam_model is a string
  std::string cam_model = calib_cam.intrinsics[0]->name();
  // loop over every landmarks
  for (auto& lm : landmarks) {
    // loop over every obs
    for (auto& lm_obs : lm.second.obs) {
      if (cameras.count(lm_obs.first) != 0) {
        // get the featureId for this camera in this landmarks
        FeatureId fid = lm_obs.second;
        // using Corners = tbb::concurrent_unordered_map<TimeCamId,
        // KeypointsData>;
        Eigen::Vector2d p_2d =
            feature_corners.find(lm_obs.first)->second.corners[fid];

        BundleAdjustmentReprojectionCostFunctor* c =
            new BundleAdjustmentReprojectionCostFunctor(p_2d, cam_model);

        ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<
            BundleAdjustmentReprojectionCostFunctor, 2, 7, 3, 8>(c);

        if (options.use_huber == true) {
          problem.AddResidualBlock(
              cost_function, new ceres::HuberLoss(options.huber_parameter),
              cameras.find(lm_obs.first)->second.T_w_c.data(),
              lm.second.p.data(),
              calib_cam.intrinsics[lm_obs.first.cam_id]->data());
        } else {
          problem.AddResidualBlock(
              cost_function, NULL,
              cameras.find(lm_obs.first)->second.T_w_c.data(),
              lm.second.p.data(),
              calib_cam.intrinsics[lm_obs.first.cam_id]->data());
        }
      }
    }
  }
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
}  // bundle_adjustment

// Run bundle adjustment to optimize cameras, points, and optionally
// intrinsics
void local_bundle_adjustment(const Corners& feature_corners,
                             const BundleAdjustmentOptions& options,
                             Calibration& calib_cam, Cameras& fixed_cameras,
                             Cameras& activate_cameras, Cameras& cameras,
                             Landmarks& landmarks) {
  ceres::Problem problem;

  // Adding parameter T_w_c for every camera
  for (auto& ca : activate_cameras) {
    // std::cout << " add cameras " << ca.first << std::endl;
    problem.AddParameterBlock(ca.second.T_w_c.data(),
                              ca.second.T_w_c.num_parameters,
                              new Sophus::test::LocalParameterizationSE3);
  }
  for (auto& ca : fixed_cameras) {
    // std::cout << " add cameras " << ca.first << std::endl;

    problem.AddParameterBlock(ca.second.T_w_c.data(),
                              ca.second.T_w_c.num_parameters,
                              new Sophus::test::LocalParameterizationSE3);
    // if this camrera is in the set of fixed cameras, set parameter to be
    // constant
    problem.SetParameterBlockConstant(ca.second.T_w_c.data());
  }

  // adding intrinsics parameters use maximum number 8
  for (auto& intr : calib_cam.intrinsics) {
    problem.AddParameterBlock(intr->data(), 8);
    // optimizing intrinstics or not depending on options
    if (options.optimize_intrinsics == false) {
      problem.SetParameterBlockConstant(intr->data());
    }
  }

  // get the name of camera model, cam_model is a string
  std::string cam_model = calib_cam.intrinsics[0]->name();
  // loop over every landmarks
  for (auto& lm : landmarks) {
    for (auto& lm_outliers : lm.second.outlier_obs) {
      // if landmark is an outliers of activate cameras, don't add this landmark
      // in local BA
      if (activate_cameras.count(lm_outliers.first) != 0) {
        continue;
      }
    }
    // loop over every obs
    for (auto& lm_obs : lm.second.obs) {
      if (activate_cameras.count(lm_obs.first) != 0) {
        // get the featureId for this camera in this landmarks
        FeatureId fid = lm_obs.second;
        // using Corners = tbb::concurrent_unordered_map<TimeCamId,
        // KeypointsData>;
        Eigen::Vector2d p_2d =
            feature_corners.find(lm_obs.first)->second.corners[fid];

        BundleAdjustmentReprojectionCostFunctor* c =
            new BundleAdjustmentReprojectionCostFunctor(p_2d, cam_model);

        ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<
            BundleAdjustmentReprojectionCostFunctor, 2, 7, 3, 8>(c);

        if (options.use_huber == true) {
          problem.AddResidualBlock(
              cost_function, new ceres::HuberLoss(options.huber_parameter),
              cameras.find(lm_obs.first)->second.T_w_c.data(),
              lm.second.p.data(),
              calib_cam.intrinsics[lm_obs.first.cam_id]->data());
        } else {
          problem.AddResidualBlock(
              cost_function, NULL,
              cameras.find(lm_obs.first)->second.T_w_c.data(),
              lm.second.p.data(),
              calib_cam.intrinsics[lm_obs.first.cam_id]->data());
        }
      }
    }
  }
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
}  // local_bundle_adjustment

// optimize relative pose for loop
void relative_se3_optimization(const TimeCamId& tcidi, const TimeCamId& tcidl,
                               const Cameras& cameras,
                               const Corners& feature_corners,
                               Calibration& calib_cam, Landmarks& landmarks,
                               MatchData& md) {
  ceres::Problem problem;
  Sophus::SE3d T_i_l = md.T_i_j;

  //  Setup optimization problem

  problem.AddParameterBlock(T_i_l.data(), T_i_l.num_parameters,
                            new Sophus::test::LocalParameterizationSE3);

  std::vector<Eigen::Vector2d> pi_2ds;
  std::vector<Eigen::Vector2d> pl_2ds;
  std::vector<Eigen::Vector3d> pi_3ds;
  std::vector<Eigen::Vector3d> pl_3ds;

  double huber_parameter = 1.0;
  int max_num_iterations = 50;

  for (auto& inlier : md.inliers) {
    FeatureId featurei = inlier.first;
    FeatureId featurel = inlier.second;
    Eigen::Vector2d pi_2d =
        feature_corners.find(tcidi)->second.corners[featurei];
    Eigen::Vector2d pl_2d =
        feature_corners.find(tcidl)->second.corners[featurel];
    Eigen::Vector3d pi_3d;
    Eigen::Vector3d pl_3d;
    pi_3d.setZero();
    pl_3d.setZero();
    for (const auto& lm : landmarks) {
      if (lm.second.obs.count(tcidi) != 0) {
        FeatureId featuretracki = lm.second.obs.find(tcidi)->second;
        if (featuretracki == featurei) {
          pi_3d = cameras.find(tcidi)->second.T_w_c.inverse() * lm.second.p;
        }
      }
      if (lm.second.obs.count(tcidl) != 0) {
        FeatureId featuretrackl = lm.second.obs.find(tcidl)->second;
        if (featuretrackl == featurel) {
          pl_3d = cameras.find(tcidl)->second.T_w_c.inverse() * lm.second.p;
        }
      }
    }
    Eigen::Vector3d p0;
    p0.setZero();
    if (pi_3d == p0 || pl_3d == p0) {
      continue;
    } else {
      pi_2ds.push_back(pi_2d);
      pl_2ds.push_back(pl_2d);
      pi_3ds.push_back(pi_3d);
      pl_3ds.push_back(pl_3d);
    }
  }
  // adding intrinsics parameters use maximum number 8
  for (auto& intr : calib_cam.intrinsics) {
    problem.AddParameterBlock(intr->data(), 8);
    // not optimizing intrinstics
    problem.SetParameterBlockConstant(intr->data());
  }

  // get the name of camera model, cam_model is a string
  std::string cam_model = calib_cam.intrinsics[0]->name();
  // loop over every landmarks
  std::cout << "pi_2ds.size" << pi_2ds.size() << std::endl;
  if (pi_2ds.size() > 8) {
    for (size_t i = 0; i < pi_2ds.size(); i++) {
      Eigen::Vector2d pi_2d = pi_2ds[i];
      Eigen::Vector3d pi_3d = pi_3ds[i];
      Eigen::Vector2d pl_2d = pl_2ds[i];
      Eigen::Vector3d pl_3d = pl_3ds[i];

      RelativeSim3ReprojectionCostFunctor* c =
          new RelativeSim3ReprojectionCostFunctor(pi_2d, pi_3d, pl_2d, pl_3d,
                                                  cam_model);

      ceres::CostFunction* cost_function =
          new ceres::AutoDiffCostFunction<RelativeSim3ReprojectionCostFunctor,
                                          4, 7, 8>(c);
      // use huber loss function
      problem.AddResidualBlock(
          cost_function, new ceres::HuberLoss(huber_parameter), T_i_l.data(),
          calib_cam.intrinsics[tcidi.cam_id]->data());
    }

    UNUSED(feature_corners);
    UNUSED(calib_cam);

    // Solve
    ceres::Solver::Options ceres_options;
    ceres_options.max_num_iterations = max_num_iterations;
    // For small problems like that better use ceres::DENSE_QR
    ceres_options.linear_solver_type = ceres::DENSE_QR;
    ceres_options.num_threads = tbb::task_scheduler_init::default_num_threads();
    ceres::Solver::Summary summary;
    Solve(ceres_options, &problem, &summary);
    md.T_i_j = T_i_l;

    std::cout << summary.BriefReport() << std::endl;
  }

}  // namespace visnav

// optimize relative pose for loop
void relative_se3_optimization_using_lmmatches(
    const TimeCamId& tcidi, const TimeCamId& tcidl, const Cameras& cameras,
    const Corners& feature_corners, Calibration& calib_cam,
    Landmarks& landmarks, LandmarkMatchData& md_i, LandmarkMatchData& md_l,
    Sophus::SE3d& T_i_l) {
  ceres::Problem problem;

  //  Setup optimization problem

  problem.AddParameterBlock(T_i_l.data(), T_i_l.num_parameters,
                            new Sophus::test::LocalParameterizationSE3);

  std::vector<Eigen::Vector2d> pi_2ds;
  std::vector<Eigen::Vector2d> pl_2ds;
  std::vector<Eigen::Vector3d> pi_3ds;
  std::vector<Eigen::Vector3d> pl_3ds;

  double huber_parameter = 1.0;
  int max_num_iterations = 50;

  for (auto& mt : md_i.matches) {
    FeatureId featurei = mt.first;
    TrackId tid_l = mt.second;
    Eigen::Vector2d pi_2d =
        feature_corners.find(tcidi)->second.corners[featurei];
    Eigen::Vector3d pl_3d = cameras.find(tcidl)->second.T_w_c.inverse() *
                            landmarks.find(tid_l)->second.p;
    pi_2ds.push_back(pi_2d);
    pl_3ds.push_back(pl_3d);
  }

  for (auto& mt : md_l.matches) {
    FeatureId featurel = mt.first;
    TrackId tid_i = mt.second;
    Eigen::Vector2d pl_2d =
        feature_corners.find(tcidl)->second.corners[featurel];

    Eigen::Vector3d pi_3d = cameras.find(tcidi)->second.T_w_c.inverse() *
                            landmarks.find(tid_i)->second.p;

    pl_2ds.push_back(pl_2d);
    pi_3ds.push_back(pi_3d);
  }
  // adding intrinsics parameters use maximum number 8
  for (auto& intr : calib_cam.intrinsics) {
    problem.AddParameterBlock(intr->data(), 8);
    // not optimizing intrinstics
    problem.SetParameterBlockConstant(intr->data());
  }

  // get the name of camera model, cam_model is a string
  std::string cam_model = calib_cam.intrinsics[0]->name();
  // loop over every landmarks

  if (pi_2ds.size() > 8 && pl_2ds.size() > 8) {
    for (size_t i = 0; i < std::min(pi_2ds.size(), pl_2ds.size()); i++) {
      Eigen::Vector2d pi_2d = pi_2ds[i];
      Eigen::Vector3d pi_3d = pi_3ds[i];
      Eigen::Vector2d pl_2d = pl_2ds[i];
      Eigen::Vector3d pl_3d = pl_3ds[i];

      RelativeSim3ReprojectionCostFunctor* c =
          new RelativeSim3ReprojectionCostFunctor(pi_2d, pi_3d, pl_2d, pl_3d,
                                                  cam_model);

      ceres::CostFunction* cost_function =
          new ceres::AutoDiffCostFunction<RelativeSim3ReprojectionCostFunctor,
                                          4, 7, 8>(c);
      // use huber loss function
      problem.AddResidualBlock(
          cost_function, new ceres::HuberLoss(huber_parameter), T_i_l.data(),
          calib_cam.intrinsics[tcidi.cam_id]->data());
    }

    UNUSED(feature_corners);
    UNUSED(calib_cam);

    // Solve
    ceres::Solver::Options ceres_options;
    ceres_options.max_num_iterations = max_num_iterations;
    // For small problems like that better use ceres::DENSE_QR
    ceres_options.linear_solver_type = ceres::DENSE_QR;
    ceres_options.num_threads = tbb::task_scheduler_init::default_num_threads();
    ceres::Solver::Summary summary;
    Solve(ceres_options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;
  }

}  // namespace visnav

void update_active_cameras(const TimeCamId& tcidl, const TimeCamId& tcidr,
                           Cameras& active_cameras, const Cameras& cameras,
                           const CovisibilityGraph& covisibility_graph,
                           const int threshold) {
  active_cameras.clear();

  // add the cameras in current frame
  active_cameras.insert(*cameras.find(tcidl));
  active_cameras.insert(*cameras.find(tcidr));

  std::set<FrameId> neighbors;
  neighbors = return_neighbors_in_covisibility(covisibility_graph, tcidl.t_ns,
                                               threshold);
  for (const auto& nb : neighbors) {
    // if this camera doesn't exist in current active_cameras, avoiding adding
    // the same camera twice
    TimeCamId tidl(nb, 0);
    if (active_cameras.count(tidl) == 0 && nb != 0) {
      TimeCamId tidr(nb, 1);

      // add left camera
      active_cameras.insert(*cameras.find(tidl));
      // add right camera
      active_cameras.insert(*cameras.find(tidr));
    }
  }

  // make sure the active cameras have mindesten 10 keyframe (20 cameras)
  for (auto rit = cameras.rbegin(); rit != cameras.rend(); ++rit) {
    if (active_cameras.count(rit->first) == 0) {
      active_cameras.insert(*rit);
    }
    if (active_cameras.size() >= 20) {
      break;
    }
  }
}
void update_active_landmarks(const Cameras& active_cameras,
                             const Landmarks& landmarks,
                             Landmarks& active_landmarks) {
  active_landmarks.clear();
  for (const auto& ca : active_cameras) {
    for (const auto& lm : landmarks) {
      // if this camera observe this landmark
      if (lm.second.obs.count(ca.first)) {
        // if this landmark doesn't exist in current active_landmarks,
        // avoiding adding the same landmark twice
        if (active_landmarks.count(lm.first) == 0) {
          active_landmarks.insert(lm);
        }
      }
    }
  }
}

void update_fixed_cameras(const Cameras& cameras, const Cameras& active_cameras,
                          Cameras& fixed_cameras, Landmarks& active_landmarks) {
  fixed_cameras.clear();
  // the first frame always fixed
  TimeCamId tid(0, 0);
  fixed_cameras.insert(*cameras.find(tid));
  tid.cam_id = 1;
  fixed_cameras.insert(*cameras.find(tid));

  // All other keyframes that see those points but are not connected to the
  // currently processed keyframe are included in the optimization but remain
  // fixed
  for (const auto& lm : active_landmarks) {
    for (const auto& obs : lm.second.obs) {
      // if this camera is not in the active_camera and it didn't be added
      // before
      if (active_cameras.count(obs.first) == 0 &&
          fixed_cameras.count(obs.first) == 0) {
        fixed_cameras.insert(*cameras.find(obs.first));
        TimeCamId tid_stereo = obs.first;
        if (obs.first.cam_id == 0) {
          tid_stereo.cam_id = 1;
        } else {
          tid_stereo.cam_id = 0;
        }
        if (active_cameras.count(tid_stereo) == 0 &&
            fixed_cameras.count(tid_stereo) == 0) {
          fixed_cameras.insert(*cameras.find(tid_stereo));
        }
      }
    }
  }
}

}  // namespace visnav
