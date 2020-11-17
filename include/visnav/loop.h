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

#include <visnav/common_types.h>

#include <visnav/bow_db.h>
#include <visnav/bow_voc.h>
#include <visnav/calibration.h>
#include <visnav/covisibility_graph.h>
#include <visnav/keypoints.h>
#include <visnav/map_utils.h>
#include <visnav/vo_utils.h>

#include <visnav/matching_utils.h>

#include <visnav/camera_models.h>
#include <set>

namespace visnav {

// calculate the diffences bewteen two bow vectors in L1 norm
double compare(const BowVector& bow_vector1, const BowVector& bow_vector2,
               const TimeCamId& tcid) {
  double diff = 0;
  /// Database for BoW lookup
  BowDatabase bow_db;

  bow_db.insert(tcid, bow_vector2);
  BowQueryResult results;

  bow_db.query(bow_vector1, 1, results);

  if (results.size() != 0) {
    diff = results.begin()->second;

  } else {
    diff = 0;
  }
  return diff;
}

// return loop candidates
bool find_Loop_Candidates(CovisibilityGraph& covisibility_gragh,
                          const TimeCamId& tcidl, const TimeCamId& tcidr,
                          Corners& feature_corners,
                          std::shared_ptr<BowVocabulary>& bow_voc,
                          const Cameras& cameras,
                          std::set<FrameId>& loop_candidates,
                          const int num_loop_candidates_same_observations) {
  std::set<FrameId> maybe_loop_candidates;
  // store the connected neighbors in covisibility graph
  std::set<FrameId> neighbors;
  // requirement

  // return all its neighbors in the covisibility graph
  neighbors = return_neighbors_in_covisibility(
      covisibility_gragh, tcidl.t_ns, num_loop_candidates_same_observations);

  // if this frame has no neightbors
  if (neighbors.size() == 0) {
    return false;
  }

  /// Database for BoW lookup for lowest score in neighbors.
  BowDatabase bow_db_neighbors;

  // get the bow vector representation of this frame, left camera
  BowVector v_tcidl;
  bow_voc->transform(feature_corners.find(tcidl)->second.corner_descriptors,
                     v_tcidl);

  // get the bow vector representation of this frame, right camera
  BowVector v_tcidr;
  bow_voc->transform(feature_corners.find(tcidr)->second.corner_descriptors,
                     v_tcidr);

  // compute the similarity between the bag of words vector of current frame and
  // all its neighbors in the covisibility graph and retain the lowest score
  double largest_difference_left = 0;
  double largest_difference_right = 0;

  // for all the neighbors from current frame
  for (const auto& nb : neighbors) {
    // get the bow vector representation of all neighbors of this frame
    TimeCamId tcid_nb_left(nb, 0);
    TimeCamId tcid_nb_right(nb, 1);

    BowVector v_nb_left;
    bow_voc->transform(
        feature_corners.find(tcid_nb_left)->second.corner_descriptors,
        v_nb_left);

    BowVector v_nb_right;
    bow_voc->transform(
        feature_corners.find(tcid_nb_left)->second.corner_descriptors,
        v_nb_right);

    // compute the similarity between the bag of words vector of current frame
    // and all its neighbors
    double difference_left;
    double difference_right;

    difference_left = compare(v_tcidl, v_nb_left, tcid_nb_left);
    difference_right = compare(v_tcidr, v_nb_right, tcid_nb_right);

    // get the largest difference
    if (difference_left > largest_difference_left) {
      largest_difference_left = difference_left;
    }
    if (difference_right > largest_difference_right) {
      largest_difference_right = difference_right;
    }
  }

  std::map<TimeCamId, bool> if_camera_in_neighbors;
  // first set all the cameras to be false
  for (const auto& ca : cameras) {
    if_camera_in_neighbors.insert(std::make_pair(ca.first, false));
  }

  for (const auto& nb : neighbors) {
    TimeCamId tcid_nb_left(nb, 0);
    TimeCamId tcid_nb_right(nb, 1);
    // set the neighbors to be true
    if_camera_in_neighbors.find(tcid_nb_left)->second = true;
    if_camera_in_neighbors.find(tcid_nb_right)->second = true;
  }

  for (const auto& ca : cameras) {
    // if it is not the same frame as the current frame and it is the left
    // timeid
    if (ca.first != tcidl && ca.first.cam_id == tcidl.cam_id &&
        if_camera_in_neighbors.find(ca.first)->second == false) {
      double diff_left = 0;
      double diff_right = 0;

      // get the bow vector for this frame, left camera
      BowVector v_ca_left;
      bow_voc->transform(
          feature_corners.find(ca.first)->second.corner_descriptors, v_ca_left);

      diff_left = compare(v_tcidl, v_ca_left, ca.first);

      // get the bow vector for this frame, right camera
      BowVector v_ca_right;
      TimeCamId ca_right(ca.first.t_ns, 1);
      bow_voc->transform(
          feature_corners.find(ca_right)->second.corner_descriptors,
          v_ca_right);

      diff_right = compare(v_tcidr, v_ca_right, ca_right);

      // if camera is not the neighbors of current frame, all those keyframes
      // directly connected to Ki are discarded from the results.

      // std::cout << ca.first << "diff:" << diff_left
      //          << " lowest score: " << largest_difference_left << std::endl;
      if (diff_left < largest_difference_left ||
          diff_right < largest_difference_right) {
        maybe_loop_candidates.insert(ca.first.t_ns);
        // std::cout << "maybe loop candidate: " << ca.first << std::endl;
      }
    }
  }

  // To accept a loop candidate we must detect consecutively three loop
  // candidates that are consistent
  for (auto it = maybe_loop_candidates.begin();
       it != maybe_loop_candidates.end();) {
    std::set<FrameId> it_neighbors;
    it_neighbors = return_neighbors_in_covisibility(
        covisibility_gragh, *it, num_loop_candidates_same_observations);
    std::set<FrameId> intersection;
    std::set_intersection(maybe_loop_candidates.begin(),
                          maybe_loop_candidates.end(), it_neighbors.begin(),
                          it_neighbors.end(),
                          std::inserter(intersection, intersection.begin()));

    // if have detected three loop candidates that are consistent
    if (intersection.size() < 3) {
      // it = loop_candidates.erase(it);
      // std::cout << "maybe loop candidate not good: " << *it << std::endl;
      ++it;
    } else {
      loop_candidates.insert(*it);
      ++it;
    }
  }
  if (loop_candidates.size() > 0) {
    return true;
  } else {
    return false;
  }
}

// tcidi is the current frameId, tcidl is the loop candidates
bool if_loop_is_accepted(
    const TimeCamId& tcidi, const TimeCamId& tcidl, Corners& feature_corners,
    Calibration& calib_cam, MatchData& md, Landmarks& landmarks,
    Cameras& cameras, const int point_cloud_ransac_min_inliers,
    const double point_cloud_ransac_thresh, const int num_ORB_feature_matches,
    const double cam_z_threshold, const int feature_match_max_dist,
    const double match_max_dist_2d, const double feature_match_test_next_best) {
  md.matches.clear();
  const KeypointsData& fi = feature_corners.find(tcidi)->second;
  const KeypointsData& fl = feature_corners.find(tcidl)->second;
  std::vector<std::bitset<256>> descriptors_i = fi.corner_descriptors;
  std::vector<std::bitset<256>> descriptors_l = fl.corner_descriptors;

  std::vector<std::bitset<256>> descriptors_i_selected;
  std::vector<std::bitset<256>> descriptors_l_selected;
  std::vector<FeatureId> featuredid_i;
  std::vector<FeatureId> featuredid_l;
  for (const auto& lm : landmarks) {
    // if the landmarks is observed by current frame
    if (lm.second.obs.count(tcidi) != 0) {
      FeatureId fid = lm.second.obs.find(tcidi)->second;
      descriptors_i_selected.push_back(descriptors_i[fid]);
      featuredid_i.push_back(fid);
    }
    // if the landmarks is observed by loop candidates
    if (lm.second.obs.count(tcidl) != 0) {
      FeatureId fid = lm.second.obs.find(tcidl)->second;
      descriptors_l_selected.push_back(descriptors_l[fid]);
      featuredid_l.push_back(fid);
    }
  }

  MatchData md_temp;
  // compute the correspendences between ORB associated with landmarks in the
  // current keyframe and the loop candidates keyframes
  matchDescriptors(descriptors_i_selected, descriptors_l_selected,
                   md_temp.matches, feature_match_max_dist,
                   feature_match_test_next_best);

  for (const auto& mt : md_temp.matches) {
    md.matches.push_back(
        std::make_pair(featuredid_i[mt.first], featuredid_l[mt.second]));
  }

  std::cout << "ORB feature matches size" << md.matches.size() << std::endl;

  // perform RANSAC iterations with each candidate, 3D-3D, trying to find a
  // similarity transformation using the method of Horn
  if (int(md.matches.size()) >= num_ORB_feature_matches) {
    ThreeDRansac(landmarks, cameras, tcidi, tcidl, point_cloud_ransac_thresh,
                 point_cloud_ransac_min_inliers, md);
  } else {
    return false;
  }

  std::cout << "after ransac md.T_i_j" << md.T_i_j.matrix() << std::endl;

  std::cout << "inliers size after RANSAN" << md.inliers.size() << std::endl;

  // if the transform is supported by enough inliers, we optimize it
  if (int(md.inliers.size()) >= point_cloud_ransac_min_inliers) {
    relative_se3_optimization(tcidi, tcidl, cameras, feature_corners, calib_cam,
                              landmarks, md);
  } else {
    return false;
  }

  std::cout << "after se3 opt md.T_i_j" << md.T_i_j.matrix() << std::endl;

  // Sophus::SE3d T_i_l = md.T_i_j;
  // To do: perform a guided search of more correspondences.
  Landmarks landmarks_obeserved_by_i;
  Landmarks landmarks_obeserved_by_l;

  for (const auto& lm : landmarks) {
    // if this landmark is observed by current frame
    if (lm.second.obs.count(tcidi) != 0) {
      landmarks_obeserved_by_i.insert(lm);
    }
    // if this landmark is observed by loop candidates frame
    if (lm.second.obs.count(tcidl) != 0) {
      landmarks_obeserved_by_l.insert(lm);
    }
  }

  // project the landmarks that observed by current frame into loop candidates
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      projected_points_l;
  // store the track id of landmarks
  std::vector<TrackId> projected_track_ids_l;
  //<FeatureId, TrackId>
  // matches between loop candidates detected points and projected landmarks
  // that observed by current frame
  LandmarkMatchData md_l;

  // for projecting landmarks observed by tcidi tino tcidl, you first convert
  // the 3d point into tcidi frame (using tcidi's world pose), then you convert
  // them to tcidl frame using T_i_j, and then you project.
  project_landmarks(cameras[tcidi].T_w_c * md.T_i_j, calib_cam.intrinsics[0],
                    landmarks_obeserved_by_i, cam_z_threshold,
                    projected_points_l, projected_track_ids_l);

  // find the matches bewteen detectd points and landmarks
  find_matches_landmarks(feature_corners[tcidl], landmarks, feature_corners,
                         projected_points_l, projected_track_ids_l,
                         match_max_dist_2d, feature_match_max_dist,
                         feature_match_test_next_best, md_l);

  std::cout << "md_l.matches.size(): " << md_l.matches.size() << std::endl;

  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      projected_points_i;
  std::vector<TrackId> projected_track_ids_i;

  project_landmarks(cameras[tcidl].T_w_c * md.T_i_j.inverse(),
                    calib_cam.intrinsics[0], landmarks_obeserved_by_l,
                    cam_z_threshold, projected_points_i, projected_track_ids_i);

  LandmarkMatchData md_i;

  // find the matches bewteen detectd points and landmarks
  find_matches_landmarks(feature_corners[tcidi], landmarks, feature_corners,
                         projected_points_i, projected_track_ids_i,
                         match_max_dist_2d, feature_match_max_dist,
                         feature_match_test_next_best, md_i);

  std::cout << "md_i.matches.size(): " << md_i.matches.size() << std::endl;

  // we optimize it again
  relative_se3_optimization_using_lmmatches(tcidi, tcidl, cameras,
                                            feature_corners, calib_cam,
                                            landmarks, md_i, md_l, md.T_i_j);

  std::cout << "after more correspondeces opt md.T_i_j:" << md.T_i_j.matrix()
            << std::endl;

  // if translation is supported by enough inliers, the loop is accepted
  if (int(md.inliers.size()) >= point_cloud_ransac_min_inliers) {
    return true;
  } else {
    return false;
  }
  UNUSED(cam_z_threshold);
  UNUSED(feature_match_max_dist);
  UNUSED(feature_match_test_next_best);
  UNUSED(match_max_dist_2d);
  UNUSED(cameras);

}  // namespace visnav

// tcidl is the current left frame,tcidr is the current right frame
void correct_current_frame_and_its_neighbors(
    const TimeCamId& tcidl, const TimeCamId& tcidr,
    const TimeCamId& tcid_loop_left, const TimeCamId& tcid_loop_right,
    Cameras& cameras, MatchData& md, CovisibilityGraph& covisibility_graph) {
  // get the ridig body translation
  Sophus::SE3d T_i_l = md.T_i_j;
  Sophus::SE3d T_w_i = cameras.find(tcidl)->second.T_w_c;
  // get all the neighbors of current frame from covisibility graph
  std::set<FrameId> tcidl_neighbors;
  tcidl_neighbors =
      return_neighbors_in_covisibility(covisibility_graph, tcidl.t_ns, 0);
  // correct the current keyframe pose
  std::cout << "correct frame: " << tcidl.t_ns << std::endl;

  cameras.find(tcidl)->second.T_w_c =
      cameras.find(tcid_loop_left)->second.T_w_c * T_i_l.inverse();
  cameras.find(tcidr)->second.T_w_c =
      cameras.find(tcid_loop_right)->second.T_w_c * T_i_l.inverse();
  /*
  cameras.find(tcidl)->second.T_w_c =
      T_i_l * cameras.find(tcid_loop_left)->second.T_w_c;
  cameras.find(tcidr)->second.T_w_c =
      T_i_l * cameras.find(tcid_loop_right)->second.T_w_c;*/
  // correct the pose of the neighbors of current frame
  for (const auto& nb : tcidl_neighbors) {
    std::cout << "correct neighbor frame: " << nb << std::endl;
    TimeCamId tid_left(nb, 0);
    TimeCamId tid_right(nb, 1);
    /*
    cameras.find(tid_left)->second.T_w_c = cameras.find(tcidl)->second.T_w_c *
                                           T_w_i.inverse() *
                                           cameras.find(tid_left)->second.T_w_c;
    cameras.find(tid_right)->second.T_w_c =
        cameras.find(tcidl)->second.T_w_c * T_w_i.inverse() *
        cameras.find(tid_right)->second.T_w_c;*/
    cameras.find(tid_left)->second.T_w_c = cameras.find(tcidl)->second.T_w_c *
                                           T_w_i.inverse() *
                                           cameras.find(tid_left)->second.T_w_c;
    cameras.find(tid_right)->second.T_w_c =
        cameras.find(tcidl)->second.T_w_c * T_w_i.inverse() *
        cameras.find(tid_right)->second.T_w_c;
  }
}

void get_to_be_projected_landmarks(const TimeCamId& tcidl, Landmarks& landmarks,
                                   Landmarks& lm_to_be_projected,
                                   CovisibilityGraph& covisibility_graph) {
  // All map points seen by the loop keyframe and its neighbors are projected
  // into current frame and its neighbors

  // get all the neighbors of loop keyframes from covisibility graph
  std::set<FrameId> tcid_neighbors;
  tcid_neighbors =
      return_neighbors_in_covisibility(covisibility_graph, tcidl.t_ns, 0);
  // add itself
  tcid_neighbors.insert(tcidl.t_ns);
  for (const auto& lm : landmarks) {
    for (const auto nb : tcid_neighbors) {
      TimeCamId tid_left(nb, 0);
      TimeCamId tid_right(nb, 1);
      if (lm.second.obs.count(tid_left) || lm.second.obs.count(tid_right)) {
        // if this landmark didn't be added before
        if (lm_to_be_projected.count(lm.first) == 0) {
          lm_to_be_projected.insert(lm);
        }
      }
    }
  }
}

void fuse_landmarks(const TimeCamId& tcidl, const TimeCamId& tcid_loop_left,
                    Landmarks& landmarks, const Cameras& cameras,
                    CovisibilityGraph& covisibility_graph,
                    Corners& feature_corners, Calibration& calib_cam,
                    std::map<TrackId, TrackId>& lm_lm_inliers,
                    LandmarkMatchData& md, const double match_max_dist_2d,
                    const int feature_match_max_dist,
                    const double feature_match_test_next_best,
                    const double cam_z_threshold) {
  // lm_lm_inliers store the landmarks TrackId correspondenced
  // <current frame, loop candidates>

  // md <FeatureId, TrackId> of current frame

  // All map points seen by the loop keyframe and its neighbors
  Landmarks lm_to_be_projected;
  get_to_be_projected_landmarks(tcid_loop_left, landmarks, lm_to_be_projected,
                                covisibility_graph);
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      projected_landmarks_loop;
  std::vector<TrackId> projected_track_ids_loop;

  // get all the neighbors of current keyframe from covisibility graph
  std::set<FrameId> tcid_neighbors;
  tcid_neighbors =
      return_neighbors_in_covisibility(covisibility_graph, tcidl.t_ns, 0);
  tcid_neighbors.insert(tcidl.t_ns);

  // duplicated landmarks correspondence <loop,current frame>
  std::vector<std::pair<TrackId, TrackId>> landmarks_correspondence;

  for (auto& nb : tcid_neighbors) {
    TimeCamId tid_left(nb, 0);
    TimeCamId tid_right(nb, 1);
    // project landmarks to the current frame and its neighbors
    project_landmarks(cameras.find(tid_left)->second.T_w_c,
                      calib_cam.intrinsics[0], lm_to_be_projected,
                      cam_z_threshold, projected_landmarks_loop,
                      projected_track_ids_loop);

    //<FeatureId, TrackId>
    LandmarkMatchData md_loop;
    KeypointsData kdl = feature_corners[tcidl];

    // md_loop find the matches bewteen detected points and the landmarks
    // obeserved from loop candidates and its neighbors
    find_matches_landmarks(kdl, landmarks, feature_corners,
                           projected_landmarks_loop, projected_track_ids_loop,
                           match_max_dist_2d, feature_match_max_dist,
                           feature_match_test_next_best, md_loop);

    for (const auto& ma : md.matches) {
      for (const auto& inl_loop : md_loop.matches) {
        // they have same FeatureId
        if (ma.first == inl_loop.first) {
          if (lm_lm_inliers.count(ma.second) == 0) {
            continue;
          } else {
            if (lm_lm_inliers.find(ma.second)->second != inl_loop.second) {
              continue;
            } else {
              // they have diffenrent TrackId
              if (inl_loop.second != ma.second) {
                landmarks_correspondence.push_back(
                    std::make_pair(inl_loop.second, ma.second));
              }
            }
          }
        }
      }
    }
  }

  for (const auto& lc : landmarks_correspondence) {
    std::cout << " duplicated landmarks correspondences: " << lc.first << " & "
              << lc.second << std::endl;
    if (lc.first != lc.second) {
      if (landmarks.count(lc.second) != 0) {
        // copy the duplicated landmarks
        FeatureTrack fids = landmarks.find(lc.second)->second.obs;
        // erase duplicated landmarks
        landmarks.erase(lc.second);
        std::cout << "delete duplicated landmarks: " << lc.second << std::endl;
        if (landmarks.count(lc.first) != 0) {
          for (auto& fid : fids) {
            landmarks.find(lc.first)->second.obs.insert(fid);
          }
        }
      }
    }
  }
}

}  // namespace visnav
