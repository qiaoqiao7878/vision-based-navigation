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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

#include <sophus/se3.hpp>

#include <tbb/concurrent_unordered_map.h>
#include <tbb/tbb.h>

#include <pangolin/display/image_view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/image/image.h>
#include <pangolin/image/image_io.h>
#include <pangolin/image/typed_image.h>
#include <pangolin/pangolin.h>

#include <CLI/CLI.hpp>

#include <visnav/common_types.h>

#include <visnav/calibration.h>

#include <visnav/keypoints.h>
#include <visnav/map_utils.h>
#include <visnav/matching_utils.h>
#include <visnav/vo_utils.h>

#include <visnav/gui_helper.h>
#include <visnav/tracks.h>

#include <visnav/serialization.h>

#include <visnav/covisibility_graph.h>
#include <visnav/essential_graph.h>

#include <visnav/bow_db.h>
#include <visnav/bow_voc.h>

#include <visnav/ground_truth_align.h>
#include <visnav/loop.h>

using namespace visnav;

///////////////////////////////////////////////////////////////////////////////
/// Declarations
///////////////////////////////////////////////////////////////////////////////

void draw_image_overlay(pangolin::View& v, size_t cam_id);
void change_display_to_image(const TimeCamId& tcid);
void draw_scene();
void load_data(const std::string& path, const std::string& calib_path);
bool next_step();
void optimize();
void compute_projections();

///////////////////////////////////////////////////////////////////////////////
/// Constants
///////////////////////////////////////////////////////////////////////////////

constexpr int UI_WIDTH = 200;
constexpr int NUM_CAMS = 2;

///////////////////////////////////////////////////////////////////////////////
/// Variables
///////////////////////////////////////////////////////////////////////////////

int num_loop = 0;

int current_frame = 0;
Sophus::SE3d current_pose;
bool take_keyframe = true;
TrackId next_landmark_id = 0;

std::atomic<bool> opt_running{false};
std::atomic<bool> opt_finished{false};

std::set<FrameId> kf_frames;

std::shared_ptr<std::thread> opt_thread;

/// intrinsic calibration
Calibration calib_cam;
Calibration calib_cam_opt;

/// loaded images
tbb::concurrent_unordered_map<TimeCamId, std::string> images;

/// timestamps for all stereo pairs
std::vector<FrameId> timestamps;

/// detected feature locations and descriptors
Corners feature_corners;

/// pairwise feature matches
Matches feature_matches;

/// camera poses in the current map
Cameras cameras;

/// camera poses in the active map for local BA
Cameras active_cameras;

/// fixed camera poses for local BA
Cameras fixed_cameras;

/// copy of cameras for optimization in parallel thread
Cameras cameras_opt;

/// copy of cameras for optimization in parallel thread
Cameras fixed_cameras_opt;

/// copy of cameras for optimization in parallel thread
Cameras active_cameras_opt;

/// loop candidates
std::set<FrameId> loop_candidates;

/// camera ground truth poses
Cameras cameras_ground_truth;

/// camera ground truth poses
Cameras cameras_ground_truth_align;

/// landmark positions and feature observations in current map
Landmarks landmarks;

/// landmark positions and feature observations in active map for local BA
Landmarks active_landmarks;

/// copy of landmarks for optimization in parallel thread
Landmarks landmarks_opt;

/// landmark positions that were removed from the current map
Landmarks old_landmarks;

/// cashed info on reprojected landmarks; recomputed every time time from
/// cameras, landmarks, and feature_tracks; used for visualization and
/// determining outliers; indexed by images
ImageProjections image_projections;

/// Vocabulary for building BoW representations.
std::shared_ptr<BowVocabulary> bow_voc;

/// Database for BoW lookup.
std::shared_ptr<BowDatabase> bow_db;

/// Covisibility Gragh
CovisibilityGraph covisibility_graph;

/// Essential Gragh
EssentialGraph essential_graph;

Sophus::SE3d T_body_cam0;

std::vector<std::pair<FrameId, FrameId>> accepted_loop;
///////////////////////////////////////////////////////////////////////////////
/// GUI parameters
///////////////////////////////////////////////////////////////////////////////

// The following GUI elements can be enabled / disabled from the main panel by
// switching the prefix from "ui" to "hidden" or vice verca. This way you can
// show only the elements you need / want for development.

pangolin::Var<bool> ui_show_hidden("ui.show_extra_options", false, true);

//////////////////////////////////////////////
/// Image display options

pangolin::Var<int> show_frame1("ui.show_frame1", 0, 0, 1500);
pangolin::Var<int> show_cam1("ui.show_cam1", 0, 0, NUM_CAMS - 1);
pangolin::Var<int> show_frame2("ui.show_frame2", 0, 0, 1500);
pangolin::Var<int> show_cam2("ui.show_cam2", 1, 0, NUM_CAMS - 1);
pangolin::Var<bool> lock_frames("ui.lock_frames", true, true);
pangolin::Var<bool> show_detected("ui.show_detected", true, true);
pangolin::Var<bool> show_matches("ui.show_matches", true, true);
pangolin::Var<bool> show_inliers("ui.show_inliers", true, true);
pangolin::Var<bool> show_reprojections("ui.show_reprojections", true, true);
pangolin::Var<bool> show_outlier_observations("ui.show_outlier_obs", false,
                                              true);
pangolin::Var<bool> show_ids("ui.show_ids", false, true);
pangolin::Var<bool> show_epipolar("hidden.show_epipolar", false, true);
pangolin::Var<bool> show_cameras3d("hidden.show_cameras", true, true);
pangolin::Var<bool> show_points3d("hidden.show_points", true, true);
pangolin::Var<bool> show_old_points3d("hidden.show_old_points3d", true, true);
pangolin::Var<bool> show_active_points3d("hidden.show_active_points3d", true,
                                         true);

pangolin::Var<bool> show_ground_truth("hidden.show_ground_truth", true, true);

pangolin::Var<bool> show_camera_trajectory("hidden.show_camera_trajectory",
                                           true, true);
pangolin::Var<bool> show_covisibility_graph("hidden.show_covisibility_graph",
                                            true, true);
pangolin::Var<bool> show_essential_graph("hidden.show_essential_graph", true,
                                         true);
pangolin::Var<bool> show_loop_closure_edge("hidden.show_loop_closure_edge",
                                           true, true);
//////////////////////////////////////////////
/// Feature extraction and matching options

pangolin::Var<int> num_features_per_image("hidden.num_features", 1500, 10,
                                          5000);
pangolin::Var<bool> rotate_features("hidden.rotate_features", true, true);
pangolin::Var<int> feature_match_max_dist("hidden.match_max_dist", 70, 1, 255);
pangolin::Var<double> feature_match_test_next_best("hidden.match_next_best",
                                                   1.2, 1, 4);

pangolin::Var<double> match_max_dist_2d("hidden.match_max_dist_2d", 20.0, 1.0,
                                        50);
// original 80
pangolin::Var<int> new_kf_min_inliers("hidden.new_kf_min_inliers", 120, 1, 200);

pangolin::Var<int> max_num_kfs("hidden.max_num_kfs", 10, 5, 20);

pangolin::Var<double> cam_z_threshold("hidden.cam_z_threshold", 0.1, 1.0, 0.0);

//////////////////////////////////////////////
/// Adding covisibility and essential graph options

// left camera observations + right camera observations
pangolin::Var<int> num_covisibility_same_observations("hidden.cov_same_obs", 20,
                                                      1, 100);
pangolin::Var<int> num_loop_candidates_same_observations("hidden.loop_same_obs",
                                                         50, 1, 100);
pangolin::Var<int> num_high_covisibility("hidden.high_cov", 50, 1, 100);

//////////////////////////////////////////////
/// deleting redundant keyframes options

pangolin::Var<int> del_kfs_num_common_keyframes("hidden.num_same_kfs(del_kfs)",
                                                3, 1, 10);
pangolin::Var<double> del_kfs_percent_shared_keyframes(
    "hidden.per_shared_kfs(del_kfs)", 0.9, 0.1, 1);

//////////////////////////////////////////////
/// Adding cameras and landmarks options

pangolin::Var<double> reprojection_error_pnp_inlier_threshold_pixel(
    "hidden.pnp_inlier_thresh", 3.0, 0.1, 10);

//////////////////////////////////////////////
/// loop options

pangolin::Var<int> point_cloud_ransac_min_inliers("hidden.3Dransac_inliers", 20,
                                                  0, 100);
pangolin::Var<double> point_cloud_ransac_thresh("hidden.3Dransac_thresh", 0.07,
                                                0, 1);
pangolin::Var<int> num_ORB_feature_matches("hidden.ORB_match", 30, 0, 100);

//////////////////////////////////////////////
/// Bundle Adjustment Options

pangolin::Var<bool> ba_optimize_intrinsics("hidden.ba_opt_intrinsics", false,
                                           true);
pangolin::Var<int> ba_verbose("hidden.ba_verbose", 1, 0, 2);

pangolin::Var<double> reprojection_error_huber_pixel("hidden.ba_huber_width",
                                                     1.0, 0.1, 10);

///////////////////////////////////////////////////////////////////////////////
/// GUI buttons
///////////////////////////////////////////////////////////////////////////////

// if you enable this, next_step is called repeatedly until completion
pangolin::Var<bool> continue_next("ui.continue_next", false, true);

using Button = pangolin::Var<std::function<void(void)>>;

Button next_step_btn("ui.next_step", &next_step);

///////////////////////////////////////////////////////////////////////////////
/// GUI and Boilerplate Implementation
///////////////////////////////////////////////////////////////////////////////

// Parse parameters, load data, and create GUI window and event loop (or
// process everything in non-gui mode).
int main(int argc, char** argv) {
  bool show_gui = true;
  std::string dataset_path = "data/V2_01_easy/mav0";
  std::string voc_path;
  std::string cam_calib = "opt_calib.json";

  CLI::App app{"SLAM."};

  app.add_option("--show-gui", show_gui, "Show GUI");
  app.add_option("--dataset-path", dataset_path,
                 "Dataset path. Default: " + dataset_path);
  app.add_option("--cam-calib", cam_calib,
                 "Path to camera calibration. Default: " + cam_calib);
  app.add_option("--voc-path", voc_path, "Vocabulary path")->required();

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  if (!voc_path.empty()) {
    bow_voc.reset(new BowVocabulary(voc_path));
    bow_db.reset(new BowDatabase);
  }

  load_data(dataset_path, cam_calib);

  if (show_gui) {
    pangolin::CreateWindowAndBind("Main", 1800, 1000);

    glEnable(GL_DEPTH_TEST);

    // main parent display for images and 3d viewer
    pangolin::View& main_view =
        pangolin::Display("main")
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0)
            .SetLayout(pangolin::LayoutEqualVertical);

    // parent display for images
    pangolin::View& img_view_display =
        pangolin::Display("images").SetLayout(pangolin::LayoutEqual);
    main_view.AddDisplay(img_view_display);

    // main ui panel
    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0,
                                          pangolin::Attach::Pix(UI_WIDTH));

    // extra options panel
    pangolin::View& hidden_panel = pangolin::CreatePanel("hidden").SetBounds(
        0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH),
        pangolin::Attach::Pix(2 * UI_WIDTH));
    ui_show_hidden.Meta().gui_changed = true;

    // 2D image views
    std::vector<std::shared_ptr<pangolin::ImageView>> img_view;
    while (img_view.size() < NUM_CAMS) {
      std::shared_ptr<pangolin::ImageView> iv(new pangolin::ImageView);

      size_t idx = img_view.size();
      img_view.push_back(iv);

      img_view_display.AddDisplay(*iv);
      iv->extern_draw_function =
          std::bind(&draw_image_overlay, std::placeholders::_1, idx);
    }

    // 3D visualization (initial camera view optimized to see full map)
    pangolin::OpenGlRenderState camera(
        pangolin::ProjectionMatrix(640, 480, 400, 400, 320, 240, 0.001, 10000),
        pangolin::ModelViewLookAt(-3.4, -3.7, -8.3, 2.1, 0.6, 0.2,
                                  pangolin::AxisNegY));

    pangolin::View& display3D =
        pangolin::Display("scene")
            .SetAspect(-640 / 480.0)
            .SetHandler(new pangolin::Handler3D(camera));
    main_view.AddDisplay(display3D);

    while (!pangolin::ShouldQuit()) {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      if (ui_show_hidden.GuiChanged()) {
        hidden_panel.Show(ui_show_hidden);
        const int panel_width = ui_show_hidden ? 2 * UI_WIDTH : UI_WIDTH;
        main_view.SetBounds(0.0, 1.0, pangolin::Attach::Pix(panel_width), 1.0);
      }

      display3D.Activate(camera);
      glClearColor(0.95f, 0.95f, 0.95f, 1.0f);  // light gray background

      draw_scene();

      img_view_display.Activate();

      if (lock_frames) {
        // in case of locking frames, chaning one should change the other
        if (show_frame1.GuiChanged()) {
          change_display_to_image(TimeCamId(show_frame1, 0));
          change_display_to_image(TimeCamId(show_frame1, 1));
        } else if (show_frame2.GuiChanged()) {
          change_display_to_image(TimeCamId(show_frame2, 0));
          change_display_to_image(TimeCamId(show_frame2, 1));
        }
      }

      if (show_frame1.GuiChanged() || show_cam1.GuiChanged()) {
        size_t frame_id = show_frame1;
        size_t cam_id = show_cam1;

        TimeCamId tcid;
        tcid.t_ns = frame_id;
        tcid.cam_id = cam_id;
        if (images.find(tcid) != images.end()) {
          pangolin::TypedImage img = pangolin::LoadImage(images[tcid]);
          img_view[0]->SetImage(img);
        } else {
          img_view[0]->Clear();
        }
      }

      if (show_frame2.GuiChanged() || show_cam2.GuiChanged()) {
        size_t frame_id = show_frame2;
        size_t cam_id = show_cam2;

        TimeCamId tcid;
        tcid.t_ns = frame_id;
        tcid.cam_id = cam_id;
        if (images.find(tcid) != images.end()) {
          pangolin::GlPixFormat fmt;
          fmt.glformat = GL_LUMINANCE;
          fmt.gltype = GL_UNSIGNED_BYTE;
          fmt.scalable_internal_format = GL_LUMINANCE8;

          pangolin::TypedImage img = pangolin::LoadImage(images[tcid]);
          img_view[1]->SetImage(img);
        } else {
          img_view[1]->Clear();
        }
      }

      pangolin::FinishFrame();

      if (continue_next) {
        // stop if there is nothing left to do
        continue_next = next_step();
      } else {
        // if the gui is just idling, make sure we don't burn too much CPU
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
    }
  } else {
    // non-gui mode: Process all frames, then exit
    while (next_step()) {
      // nop
    }
  }

  return 0;
}

// Visualize features and related info on top of the image views
void draw_image_overlay(pangolin::View& v, size_t view_id) {
  UNUSED(v);

  size_t frame_id = view_id == 0 ? show_frame1 : show_frame2;
  size_t cam_id = view_id == 0 ? show_cam1 : show_cam2;

  TimeCamId tcid(frame_id, cam_id);

  float text_row = 20;

  if (show_detected) {
    glLineWidth(1.0);
    glColor3f(1.0, 0.0, 0.0);  // red
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    if (feature_corners.find(tcid) != feature_corners.end()) {
      const KeypointsData& cr = feature_corners.at(tcid);

      for (size_t i = 0; i < cr.corners.size(); i++) {
        Eigen::Vector2d c = cr.corners[i];
        double angle = cr.corner_angles[i];
        pangolin::glDrawCirclePerimeter(c[0], c[1], 3.0);

        Eigen::Vector2d r(3, 0);
        Eigen::Rotation2Dd rot(angle);
        r = rot * r;

        pangolin::glDrawLine(c, c + r);
      }

      pangolin::GlFont::I()
          .Text("Detected %d corners", cr.corners.size())
          .Draw(5, text_row);

    } else {
      glLineWidth(1.0);

      pangolin::GlFont::I().Text("Corners not processed").Draw(5, text_row);
    }
    text_row += 20;
  }

  if (show_matches || show_inliers) {
    glLineWidth(1.0);
    glColor3f(0.0, 0.0, 1.0);  // blue
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    size_t o_frame_id = (view_id == 0 ? show_frame2 : show_frame1);
    size_t o_cam_id = (view_id == 0 ? show_cam2 : show_cam1);

    TimeCamId o_tcid(o_frame_id, o_cam_id);

    int idx = -1;

    auto it = feature_matches.find(std::make_pair(tcid, o_tcid));

    if (it != feature_matches.end()) {
      idx = 0;
    } else {
      it = feature_matches.find(std::make_pair(o_tcid, tcid));
      if (it != feature_matches.end()) {
        idx = 1;
      }
    }

    if (idx >= 0 && show_matches) {
      if (feature_corners.find(tcid) != feature_corners.end()) {
        const KeypointsData& cr = feature_corners.at(tcid);

        for (size_t i = 0; i < it->second.matches.size(); i++) {
          size_t c_idx = idx == 0 ? it->second.matches[i].first
                                  : it->second.matches[i].second;

          Eigen::Vector2d c = cr.corners[c_idx];
          double angle = cr.corner_angles[c_idx];
          pangolin::glDrawCirclePerimeter(c[0], c[1], 3.0);

          Eigen::Vector2d r(3, 0);
          Eigen::Rotation2Dd rot(angle);
          r = rot * r;

          pangolin::glDrawLine(c, c + r);

          if (show_ids) {
            pangolin::GlFont::I().Text("%d", i).Draw(c[0], c[1]);
          }
        }

        pangolin::GlFont::I()
            .Text("Detected %d matches", it->second.matches.size())
            .Draw(5, text_row);
        text_row += 20;
      }
    }

    glColor3f(0.0, 1.0, 0.0);  // green

    if (idx >= 0 && show_inliers) {
      if (feature_corners.find(tcid) != feature_corners.end()) {
        const KeypointsData& cr = feature_corners.at(tcid);

        for (size_t i = 0; i < it->second.inliers.size(); i++) {
          size_t c_idx = idx == 0 ? it->second.inliers[i].first
                                  : it->second.inliers[i].second;

          Eigen::Vector2d c = cr.corners[c_idx];
          double angle = cr.corner_angles[c_idx];
          pangolin::glDrawCirclePerimeter(c[0], c[1], 3.0);

          Eigen::Vector2d r(3, 0);
          Eigen::Rotation2Dd rot(angle);
          r = rot * r;

          pangolin::glDrawLine(c, c + r);

          if (show_ids) {
            pangolin::GlFont::I().Text("%d", i).Draw(c[0], c[1]);
          }
        }

        pangolin::GlFont::I()
            .Text("Detected %d inliers", it->second.inliers.size())
            .Draw(5, text_row);
        text_row += 20;
      }
    }
  }

  if (show_reprojections) {
    if (image_projections.count(tcid) > 0) {
      glLineWidth(1.0);
      glColor3f(1.0, 0.0, 0.0);  // red
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

      const size_t num_points = image_projections.at(tcid).obs.size();
      double error_sum = 0;
      size_t num_outliers = 0;

      // count up and draw all inlier projections
      for (const auto& lm_proj : image_projections.at(tcid).obs) {
        error_sum += lm_proj->reprojection_error;

        if (lm_proj->outlier_flags != OutlierNone) {
          // outlier point
          glColor3f(1.0, 0.0, 0.0);  // red
          ++num_outliers;
        } else if (lm_proj->reprojection_error >
                   reprojection_error_huber_pixel) {
          // close to outlier point
          glColor3f(1.0, 0.5, 0.0);  // orange
        } else {
          // clear inlier point
          glColor3f(1.0, 1.0, 0.0);  // yellow
        }
        pangolin::glDrawCirclePerimeter(lm_proj->point_reprojected, 3.0);
        pangolin::glDrawLine(lm_proj->point_measured,
                             lm_proj->point_reprojected);
      }

      // only draw outlier projections
      if (show_outlier_observations) {
        glColor3f(1.0, 0.0, 0.0);  // red
        for (const auto& lm_proj : image_projections.at(tcid).outlier_obs) {
          pangolin::glDrawCirclePerimeter(lm_proj->point_reprojected, 3.0);
          pangolin::glDrawLine(lm_proj->point_measured,
                               lm_proj->point_reprojected);
        }
      }

      glColor3f(1.0, 0.0, 0.0);  // red
      pangolin::GlFont::I()
          .Text("Average repr. error (%u points, %u new outliers): %.2f",
                num_points, num_outliers, error_sum / num_points)
          .Draw(5, text_row);
      text_row += 20;
    }
  }

  if (show_epipolar) {
    glLineWidth(1.0);
    glColor3f(0.0, 1.0, 1.0);  // bright teal
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    size_t o_frame_id = (view_id == 0 ? show_frame2 : show_frame1);
    size_t o_cam_id = (view_id == 0 ? show_cam2 : show_cam1);

    TimeCamId o_tcid(o_frame_id, o_cam_id);

    int idx = -1;

    auto it = feature_matches.find(std::make_pair(tcid, o_tcid));

    if (it != feature_matches.end()) {
      idx = 0;
    } else {
      it = feature_matches.find(std::make_pair(o_tcid, tcid));
      if (it != feature_matches.end()) {
        idx = 1;
      }
    }

    if (idx >= 0 && it->second.inliers.size() > 20) {
      Sophus::SE3d T_this_other =
          idx == 0 ? it->second.T_i_j : it->second.T_i_j.inverse();

      Eigen::Vector3d p0 = T_this_other.translation().normalized();

      int line_id = 0;
      for (double i = -M_PI_2 / 2; i <= M_PI_2 / 2; i += 0.05) {
        Eigen::Vector3d p1(0, sin(i), cos(i));

        if (idx == 0) p1 = it->second.T_i_j * p1;

        p1.normalize();

        std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
            line;
        for (double j = -1; j <= 1; j += 0.001) {
          line.emplace_back(calib_cam.intrinsics[cam_id]->project(
              p0 * j + (1 - std::abs(j)) * p1));
        }

        Eigen::Vector2d c = calib_cam.intrinsics[cam_id]->project(p1);
        pangolin::GlFont::I().Text("%d", line_id).Draw(c[0], c[1]);
        line_id++;

        pangolin::glDrawLineStrip(line);
      }
    }
  }
}

// Update the image views to a given image id
void change_display_to_image(const TimeCamId& tcid) {
  if (0 == tcid.cam_id) {
    // left view
    show_cam1 = 0;
    show_frame1 = tcid.t_ns;
    show_cam1.Meta().gui_changed = true;
    show_frame1.Meta().gui_changed = true;
  } else {
    // right view
    show_cam2 = tcid.cam_id;
    show_frame2 = tcid.t_ns;
    show_cam2.Meta().gui_changed = true;
    show_frame2.Meta().gui_changed = true;
  }
}

// Render the 3D viewer scene of cameras and points
void draw_scene() {
  const TimeCamId tcid1(show_frame1, show_cam1);
  const TimeCamId tcid2(show_frame2, show_cam2);

  const u_int8_t color_camera_current[3]{255, 0, 0};         // red
  const u_int8_t color_camera_left[3]{0, 125, 0};            // dark green
  const u_int8_t color_camera_right[3]{0, 0, 125};           // dark blue
  const u_int8_t color_points[3]{0, 0, 0};                   // black
  const u_int8_t color_old_points[3]{170, 170, 170};         // gray
  const u_int8_t color_selected_left[3]{0, 250, 0};          // green
  const u_int8_t color_selected_right[3]{0, 0, 250};         // blue
  const u_int8_t color_selected_both[3]{0, 250, 250};        // teal
  const u_int8_t color_outlier_observation[3]{250, 0, 250};  // purple
  const u_int8_t color_active_points[3]{255, 192, 203};      // pink
  const u_int8_t color_loop_candidates[3]{93, 173, 226};     // light blue
  const u_int8_t color_active_cameras[3]{255, 165, 0};       // orange

  // render cameras
  if (show_cameras3d) {
    for (const auto& cam : cameras) {
      if (active_cameras.count(cam.first) != 0) {
        render_camera(cam.second.T_w_c.matrix(), 3.0f, color_active_cameras,
                      0.1f);
        continue;
      }
      if (loop_candidates.count(cam.first.t_ns) != 0) {
        render_camera(cam.second.T_w_c.matrix(), 3.0f, color_loop_candidates,
                      0.1f);
        continue;
      }

      if (cam.first == tcid1) {
        render_camera(cam.second.T_w_c.matrix(), 3.0f, color_selected_left,
                      0.1f);
      } else if (cam.first == tcid2) {
        render_camera(cam.second.T_w_c.matrix(), 3.0f, color_selected_right,
                      0.1f);
      } else if (cam.first.cam_id == 0) {
        render_camera(cam.second.T_w_c.matrix(), 2.0f, color_camera_left, 0.1f);
      } else {
        render_camera(cam.second.T_w_c.matrix(), 2.0f, color_camera_right,
                      0.1f);
      }
    }
    render_camera(current_pose.matrix(), 2.0f, color_camera_current, 0.1f);
  }

  // render ground truth
  if (show_ground_truth) {
    for (auto it = cameras_ground_truth_align.begin();
         it != cameras_ground_truth_align.end(); ++it) {
      if (std::next(it) != cameras_ground_truth_align.end()) {
        auto it_next = std::next(it);
        // std::cout << it->second.T_w_c.matrix() << std::endl;
        render_trajectory(
            (it->second.T_w_c * T_body_cam0.inverse()).matrix(),
            (it_next->second.T_w_c * T_body_cam0.inverse()).matrix(), 2.0f,
            color_outlier_observation);
      }
    }
  }

  // render trajectory
  if (show_camera_trajectory) {
    for (auto it = cameras.begin(); it != cameras.end(); ++it) {
      if (it->first.cam_id == 0) {
        if (cameras.size() > 2) {
          if (std::next(it, 2) != cameras.end()) {
            auto it_next = std::next(it, 2);
            // std::cout << it->second.T_w_c.matrix() << std::endl;
            render_trajectory(
                (it->second.T_w_c * T_body_cam0.inverse()).matrix(),
                (it_next->second.T_w_c * T_body_cam0.inverse()).matrix(), 2.0f,
                color_selected_left);
          }
        }
      }
    }
  }

  // render covisibility graph
  if (show_covisibility_graph) {
    for (auto it = cameras.begin(); it != cameras.end(); ++it) {
      // only draw line base on left camera
      if (it->first.cam_id == 0) {
        // if there are more than two cameras
        if (cameras.size() > 2) {
          if (covisibility_graph.count(it->first.t_ns) != 0) {
            for (const auto& nb :
                 covisibility_graph.find(it->first.t_ns)->second) {
              TimeCamId tid;
              tid.t_ns = nb.first;
              tid.cam_id = 0;
              // draw it with green line
              render_trajectory(
                  (it->second.T_w_c * T_body_cam0.inverse()).matrix(),
                  (cameras.find(tid)->second.T_w_c * T_body_cam0.inverse())
                      .matrix(),
                  1.5f, color_selected_left);
            }
          }
        }
      }
    }
  }

  // render essential graph
  if (show_essential_graph) {
    // loop over all keyframe
    for (auto it = cameras.begin(); it != cameras.end(); ++it) {
      // only draw line base on left camera
      if (it->first.cam_id == 0) {
        // if there are more than two cameras
        if (cameras.size() > 2) {
          // if this keyframe is in the essential graph
          if (essential_graph.count(it->first.t_ns) != 0) {
            // for every neighbors from it
            for (const auto& nb :
                 essential_graph.find(it->first.t_ns)->second) {
              TimeCamId tid(nb.first, 0);
              // draw it with green line
              render_trajectory(
                  (it->second.T_w_c * T_body_cam0.inverse()).matrix(),
                  (cameras.find(tid)->second.T_w_c * T_body_cam0.inverse())
                      .matrix(),
                  1.5f, color_selected_left);
            }
          }
        }
      }
    }
  }

  // render loop closure edge
  if (show_loop_closure_edge) {
    for (const auto& al : accepted_loop) {
      TimeCamId tid(al.first, 0);
      TimeCamId tid_loop(al.second, 0);
      // draw it with red line
      render_trajectory(
          (cameras.find(tid)->second.T_w_c * T_body_cam0.inverse()).matrix(),
          (cameras.find(tid_loop)->second.T_w_c * T_body_cam0.inverse())
              .matrix(),
          1.5f, color_camera_current);
    }
  }

  // render points
  if (show_points3d && landmarks.size() > 0) {
    glPointSize(3.0);
    glBegin(GL_POINTS);
    for (const auto& kv_lm : landmarks) {
      const bool in_cam_1 = kv_lm.second.obs.count(tcid1) > 0;
      const bool in_cam_2 = kv_lm.second.obs.count(tcid2) > 0;

      const bool outlier_in_cam_1 = kv_lm.second.outlier_obs.count(tcid1) > 0;
      const bool outlier_in_cam_2 = kv_lm.second.outlier_obs.count(tcid2) > 0;

      if (in_cam_1 && in_cam_2) {
        glColor3ubv(color_selected_both);
      } else if (in_cam_1) {
        glColor3ubv(color_selected_left);
      } else if (in_cam_2) {
        glColor3ubv(color_selected_right);
      } else if (outlier_in_cam_1 || outlier_in_cam_2) {
        glColor3ubv(color_outlier_observation);
      } else {
        glColor3ubv(color_points);
      }

      pangolin::glVertex(kv_lm.second.p);
    }
    glEnd();
  }

  // render points
  if (show_old_points3d && old_landmarks.size() > 0) {
    glPointSize(3.0);
    glBegin(GL_POINTS);

    for (const auto& kv_lm : old_landmarks) {
      glColor3ubv(color_old_points);
      pangolin::glVertex(kv_lm.second.p);
    }
    glEnd();
  }
  // render active landmarks
  if (show_active_points3d && active_landmarks.size() > 0) {
    glPointSize(3.0);
    glBegin(GL_POINTS);

    for (const auto& kv_lm : active_landmarks) {
      glColor3ubv(color_active_points);
      pangolin::glVertex(kv_lm.second.p);
    }
    glEnd();
  }
}

// Load images, calibration, and features / matches if available
void load_data(const std::string& dataset_path, const std::string& calib_path) {
  const std::string timestams_path = dataset_path + "/cam0/data.csv";
  const std::string groundtruth_path =
      dataset_path + "/state_groundtruth_estimate0/data.csv";
  timestamps.clear();

  {
    std::ifstream times(timestams_path);

    int64_t timestamp;

    int id = 0;
    int i = 0;

    while (times) {
      std::string line;
      std::getline(times, line);

      ++i;
      // skip the first 22 images to let the camera and ground truth start at
      // the same time
      if (i <= 30) continue;

      // for dataset V1_01_easy
      // if (line.size() < 20 || line[0] == '#' || id > 2700) continue;
      if (line.size() < 20 || line[0] == '#' || id > 2700) continue;

      // std::cout << line << std::endl;
      // 1403715273262142976,1403715273262142976.png

      std::string img_name = line.substr(20, line.size() - 21);
      std::string timestamp_str = line.substr(0, 19);
      timestamp = std::stoll(timestamp_str);
      // ensure that we actually read a new timestamp (and not e.g. just newline
      // at the end of the file)
      if (times.fail()) {
        times.clear();
        std::string temp;
        times >> temp;
        if (temp.size() > 0) {
          std::cerr << "Skipping '" << temp << "' while reading times."
                    << std::endl;
        }
        continue;
      }

      timestamps.push_back(timestamp);

      for (int i = 0; i < NUM_CAMS; i++) {
        TimeCamId tcid(id, i);

        //        std::stringstream ss;
        //        ss << dataset_path << "/" << timestamp << "_" << i << ".jpg";
        //        pangolin::TypedImage img = pangolin::LoadImage(ss.str());
        //        images[tcid] = std::move(img);

        std::stringstream ss;
        ss << dataset_path << "/cam" << i << "/data/" << img_name;

        images[tcid] = ss.str();
      }

      id++;
    }

    std::cerr << "Loaded " << id << " image pairs" << std::endl;
  }

  // load ground truth data
  {
    std::ifstream groundtruths(groundtruth_path);
    int id = 0;
    // split the data by "," and store every single value in gtdata
    std::vector<std::string> gtdata;

    while (groundtruths) {
      std::string line;
      std::getline(groundtruths, line);
      // std::cout << line << std::endl;
      // we get values that split by ","
      std::string timestamp_str = line.substr(0, 19);

      // same condition as loading images
      if (line.size() < 20 || line[0] == '#' || id > int(timestamps.size())) {
        continue;
      }
      // std::cout << id << " " << std::stoll(timestamp_str) << std::endl;

      if (std::abs(std::stoll(timestamp_str) - timestamps[id]) <= 2000000) {
        // split the line by "," until end
        // while (line.find(",") != std::string::npos) {

        // split the line by "," and get the first 8 values of each line
        for (size_t i = 0; i < 8; i++) {
          std::size_t pos = line.find(",");
          gtdata.push_back(line.substr(0, pos));
          line = line.substr(pos + 1);
        }
        id++;
      }
    }
    Eigen::Vector3d world_position;
    world_position.x() = std::stod(gtdata[1]);
    world_position.y() = std::stod(gtdata[2]);
    world_position.z() = std::stod(gtdata[3]);
    Eigen::Quaternion<double> world_qua;
    world_qua.w() = std::stod(gtdata[4]);
    world_qua.x() = std::stod(gtdata[5]);
    world_qua.y() = std::stod(gtdata[6]);
    world_qua.z() = std::stod(gtdata[7]);
    Sophus::SE3d T_w_1;
    T_w_1.setQuaternion(world_qua);
    T_w_1.translation() = world_position;

    // camera0 extrinsics wrt. the body-frame.
    Eigen::Matrix<double, 3, 3> R_body_cam0;
    R_body_cam0 << 0.0148655429818, -0.999880929698, 0.00414029679422,
        0.999557249008, 0.0149672133247, 0.025715529948, -0.0257744366974,
        0.00375618835797, 0.999660727178;
    Eigen::Matrix<double, 3, 1> t_body_cam0;
    t_body_cam0 << -0.0216401454975, -0.064676986768, 0.00981073058949;
    Sophus::SE3d T_body_cam0_init(R_body_cam0, t_body_cam0);
    T_body_cam0 = T_body_cam0_init;

    for (size_t i = 0; i < gtdata.size();) {
      TimeCamId tcid;
      // tcid.t_ns = std::stoll(gtdata[i]);
      tcid.t_ns = i / 8;
      tcid.cam_id = 0;
      Eigen::Vector3d position;
      position.x() = std::stod(gtdata[i + 1]);
      position.y() = std::stod(gtdata[i + 2]);
      position.z() = std::stod(gtdata[i + 3]);
      Eigen::Quaternion<double> qua;
      qua.w() = std::stod(gtdata[i + 4]);
      qua.x() = std::stod(gtdata[i + 5]);
      qua.y() = std::stod(gtdata[i + 6]);
      qua.z() = std::stod(gtdata[i + 7]);
      Sophus::SE3d T_w_i;
      T_w_i.setQuaternion(qua);
      T_w_i.translation() = position;
      Camera ca_gt;
      ca_gt.T_w_c =
          T_body_cam0.inverse() * T_w_1.inverse() * T_w_i * T_body_cam0;
      // ca_gt.T_w_c = T_w_1.inverse() * T_w_i;

      if (tcid.t_ns == 0) {
        Sophus::SE3d identitymatrix;
        ca_gt.T_w_c = identitymatrix;
      }
      cameras_ground_truth.insert(std::make_pair(tcid, ca_gt));
      i += 8;
    }

    std::cerr << "Loaded Ground Truth" << std::endl;
    std::cout << " Ground Truth camera" << cameras_ground_truth.size()
              << std::endl;
  }

  {
    std::ifstream os(calib_path, std::ios::binary);

    if (os.is_open()) {
      cereal::JSONInputArchive archive(os);
      archive(calib_cam);
      std::cout << "Loaded camera" << std::endl;

    } else {
      std::cerr << "could not load camera calibration " << calib_path
                << std::endl;
      std::abort();
    }
  }

  show_frame1.Meta().range[1] = images.size() / NUM_CAMS - 1;
  show_frame1.Meta().gui_changed = true;
  show_frame2.Meta().range[1] = images.size() / NUM_CAMS - 1;
  show_frame2.Meta().gui_changed = true;
}

///////////////////////////////////////////////////////////////////////////////
/// Here the algorithmically interesting implementation begins
///////////////////////////////////////////////////////////////////////////////

// Execute next step in the overall odometry pipeline. Call this repeatedly
// until it returns false for automatic execution.
bool next_step() {
  std::cout << "current frame " << current_frame << std::endl;
  if (current_frame >= int(images.size()) / NUM_CAMS) return false;

  const Sophus::SE3d T_0_1 = calib_cam.T_i_c[0].inverse() * calib_cam.T_i_c[1];

  if (take_keyframe) {
    take_keyframe = false;

    TimeCamId tcidl(current_frame, 0), tcidr(current_frame, 1);

    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
        projected_points;
    std::vector<TrackId> projected_track_ids;

    project_landmarks(current_pose, calib_cam.intrinsics[0], landmarks,
                      cam_z_threshold, projected_points, projected_track_ids);

    std::cout << "KF Projected " << projected_track_ids.size() << " points."
              << std::endl;

    MatchData md_stereo;
    KeypointsData kdl, kdr;

    pangolin::ManagedImage<uint8_t> imgl = pangolin::LoadImage(images[tcidl]);
    pangolin::ManagedImage<uint8_t> imgr = pangolin::LoadImage(images[tcidr]);

    detectKeypointsAndDescriptors(imgl, kdl, num_features_per_image,
                                  rotate_features);
    detectKeypointsAndDescriptors(imgr, kdr, num_features_per_image,
                                  rotate_features);

    md_stereo.T_i_j = T_0_1;

    Eigen::Matrix3d E;
    computeEssential(T_0_1, E);

    matchDescriptors(kdl.corner_descriptors, kdr.corner_descriptors,
                     md_stereo.matches, feature_match_max_dist,
                     feature_match_test_next_best);

    findInliersEssential(kdl, kdr, calib_cam.intrinsics[0],
                         calib_cam.intrinsics[1], E, 1e-3, md_stereo);

    std::cout << "KF Found " << md_stereo.inliers.size() << " stereo-matches."
              << std::endl;

    feature_corners[tcidl] = kdl;
    feature_corners[tcidr] = kdr;
    feature_matches[std::make_pair(tcidl, tcidr)] = md_stereo;

    LandmarkMatchData md;

    // find the matches bewteen detectd points and landmarks
    find_matches_landmarks(kdl, landmarks, feature_corners, projected_points,
                           projected_track_ids, match_max_dist_2d,
                           feature_match_max_dist, feature_match_test_next_best,
                           md);

    std::cout << "KF Found " << md.matches.size() << " matches." << std::endl;
    if (md.matches.size() >= 4) {
      localize_camera(calib_cam.intrinsics[0], kdl, landmarks,
                      reprojection_error_pnp_inlier_threshold_pixel, md);
    } else {
      md.T_w_c = current_pose;
    }
    std::cout << "Found " << md.inliers.size() << " inliers." << std::endl;

    current_pose = md.T_w_c;

    cameras[tcidl].T_w_c = current_pose;
    cameras[tcidr].T_w_c = current_pose * T_0_1;

    add_new_landmarks(tcidl, tcidr, kdl, kdr, calib_cam, md_stereo, md,
                      landmarks, next_landmark_id);

    // odometry implement: simply keep the newest keyframes

    // remove_old_keyframes(tcidl, max_num_kfs, cameras, landmarks,
    // old_landmarks,
    //                     kf_frames);

    remove_redundant_keyframes(tcidl, max_num_kfs, cameras, landmarks,
                               old_landmarks, kf_frames, accepted_loop,
                               del_kfs_num_common_keyframes,
                               del_kfs_percent_shared_keyframes);

    // update the covisibility graph

    update_covisibility_graph(covisibility_graph, tcidl, tcidr, cameras,
                              landmarks, num_covisibility_same_observations);

    // print_covisibility_gragh(covisibility_graph);

    update_essential_graph(covisibility_graph, tcidl.t_ns, cameras,
                           essential_graph, num_covisibility_same_observations,
                           num_high_covisibility);

    // print_essential_gragh(essential_graph);

    // the last one is the threshold for returning the neighbors
    update_active_cameras(tcidl, tcidr, active_cameras, cameras,
                          covisibility_graph,
                          num_covisibility_same_observations);

    // for (const auto& ca : active_cameras) {
    //  std::cout << "active_cameras " << ca.first << std::endl;
    //}

    // update active landmarks for local BA
    update_active_landmarks(active_cameras, landmarks, active_landmarks);

    // for (const auto& lm : active_landmarks) {
    //  std::cout << "active_landmarks " << lm.first << std::endl;
    //}

    // update the fixed cameras set for local BA
    update_fixed_cameras(cameras, active_cameras, fixed_cameras,
                         active_landmarks);

    // for (const auto& ca : fixed_cameras) {
    //  std::cout << "fixed_cameras " << ca.first << std::endl;
    //}
    loop_candidates.clear();
    FrameId last_loop_closure = 0;
    if (accepted_loop.size() != 0) {
      last_loop_closure = accepted_loop.back().first;
    }
    std::cout << "last_loop_closure: " << last_loop_closure << std::endl;

    // if it is 10 frames later then last loop closure
    if (current_frame - last_loop_closure > 10) {
      // find loop candidates
      bool if_find_loop_candidates;
      if_find_loop_candidates = find_Loop_Candidates(
          covisibility_graph, tcidl, tcidr, feature_corners, bow_voc, cameras,
          loop_candidates, num_loop_candidates_same_observations);
      std::cout << "loop_candidates.size(): " << loop_candidates.size()
                << std::endl;

      // if some loop candidates are found
      if (if_find_loop_candidates) {
        // for every loop candidate
        for (const auto& lc : loop_candidates) {
          std::cout << "loop candidate: " << lc << std::endl;

          // MatchData <FeatureId, FeatureId>
          MatchData md_left;
          TimeCamId tcid_loop_left(lc, 0);

          // if loop is accepted for left camera
          bool loop_is_accepted_left;
          // matches bewteen the detected points from current frame and loop
          // candidates
          loop_is_accepted_left = if_loop_is_accepted(
              tcidl, tcid_loop_left, feature_corners, calib_cam, md_left,
              landmarks, cameras, point_cloud_ransac_min_inliers,
              point_cloud_ransac_thresh, num_ORB_feature_matches,
              cam_z_threshold, feature_match_max_dist, match_max_dist_2d,
              feature_match_test_next_best);

          // if loop is accepted for right camera
          MatchData md_right;
          TimeCamId tcid_loop_right(lc, 1);
          bool loop_is_accepted_right;
          loop_is_accepted_right = if_loop_is_accepted(
              tcidr, tcid_loop_right, feature_corners, calib_cam, md_right,
              landmarks, cameras, point_cloud_ransac_min_inliers,
              point_cloud_ransac_thresh, num_ORB_feature_matches,
              cam_z_threshold, feature_match_max_dist, match_max_dist_2d,
              feature_match_test_next_best);

          // if both cameras fulfill the conditions
          if (loop_is_accepted_left || loop_is_accepted_right) {
            std::cout << "loop is accepted" << std::endl;
            // add this loop pair to accepted loop
            accepted_loop.push_back(std::make_pair(current_frame, lc));

            ++num_loop;

            // md_left store the correspondenced <current frame, loop
            // candidates>
            // lm_lm_inliers store the landmarks TrackId correspondenced
            // <current frame, loop candidates>
            std::map<TrackId, TrackId> lm_lm_inliers;

            // current keyframe pose anf all its neighbors are corrected with
            // the transformation
            if (loop_is_accepted_left) {
              correct_current_frame_and_its_neighbors(
                  tcidl, tcidr, tcid_loop_left, tcid_loop_right, cameras,
                  md_left, covisibility_graph);
              get_landmark_correspondences_from_matchdata(
                  tcidl, tcid_loop_left, landmarks, md_left, lm_lm_inliers);
            } else {
              correct_current_frame_and_its_neighbors(
                  tcidl, tcidr, tcid_loop_left, tcid_loop_right, cameras,
                  md_right, covisibility_graph);
              get_landmark_correspondences_from_matchdata(
                  tcidr, tcid_loop_right, landmarks, md_right, lm_lm_inliers);
            }

            fuse_landmarks(tcidl, tcid_loop_left, landmarks, cameras,
                           covisibility_graph, feature_corners, calib_cam,
                           lm_lm_inliers, md, match_max_dist_2d,
                           feature_match_max_dist, feature_match_test_next_best,
                           cam_z_threshold);

            // update active landmarks
            update_active_landmarks(active_cameras, landmarks,
                                    active_landmarks);

            update_covisibility_graph(covisibility_graph, tcidl, tcidr, cameras,
                                      landmarks,
                                      num_covisibility_same_observations);

            print_covisibility_gragh(covisibility_graph);

            // add_loop_closure_edge_covisibility_gragh(
            //    current_frame, tcid_loop_left.t_ns, covisibility_graph, 50);

            update_essential_graph(
                covisibility_graph, tcidl.t_ns, cameras, essential_graph,
                num_covisibility_same_observations, num_high_covisibility);

            // add_loop_closure_edge_essential_gragh(
            //    current_frame, tcid_loop_left.t_ns, essential_graph, 50);

            print_essential_gragh(essential_graph);

            // To do:pose graph optimiation over essential graph

            // cameras[tcidl].T_w_c = current_pose;
            // cameras[tcidr].T_w_c = current_pose * T_0_1;

            FrameId fid = *(kf_frames.begin());
            // std::cout << "fid " << fid << std::endl;
            std::set<TimeCamId> fixed_cameras_ba = {{fid, 0}, {fid, 1}};

            // Prepare bundle adjustment
            BundleAdjustmentOptions ba_options;
            ba_options.optimize_intrinsics = ba_optimize_intrinsics;
            ba_options.use_huber = true;
            ba_options.huber_parameter = reprojection_error_huber_pixel;
            ba_options.max_num_iterations = 100;
            ba_options.verbosity_level = ba_verbose;

            bundle_adjustment(feature_corners, ba_options, fixed_cameras_ba,
                              calib_cam, cameras, landmarks);

            // don't calculate other candidates anymore
            break;
          }
        }
      } else {
        std::cout << "no loop candidates were found!" << std::endl;
      }
    }

    std::cout << "loop number:" << num_loop << std::endl;
    for (const auto& lo : accepted_loop) {
      std::cout << "accepted loop: " << lo.first << " and " << lo.second
                << std::endl;
    }
    //

    optimize();

    current_pose = cameras[tcidl].T_w_c;

    // cameras_ground_truth was loaded at the very beginning and remain
    // unchanged
    cameras_ground_truth_align = cameras_ground_truth;
    if (cameras.size() >= 6) {
      ErrorMetricValue ate;
      // transformation that aligns the 3D points in model to ground truth
      Sophus::SE3d T_gt_camera =
          align_cameras_se3(cameras_ground_truth, cameras, &ate);
      for (auto& cam_gt : cameras_ground_truth_align) {
        cam_gt.second.T_w_c = T_gt_camera.inverse() * cam_gt.second.T_w_c;
      }
      std::cout << "rmse error: " << ate.rmse << std::endl;
    }
    // update image views
    change_display_to_image(tcidl);
    change_display_to_image(tcidr);

    compute_projections();

    current_frame++;

    return true;
  } else {
    TimeCamId tcidl(current_frame, 0), tcidr(current_frame, 1);

    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
        projected_points;
    std::vector<TrackId> projected_track_ids;
    // original: landmarks instead of active_landmarks
    project_landmarks(current_pose, calib_cam.intrinsics[0], active_landmarks,
                      cam_z_threshold, projected_points, projected_track_ids);

    std::cout << "Projected " << projected_track_ids.size() << " points."
              << std::endl;

    KeypointsData kdl;

    pangolin::ManagedImage<uint8_t> imgl = pangolin::LoadImage(images[tcidl]);

    detectKeypointsAndDescriptors(imgl, kdl, num_features_per_image,
                                  rotate_features);

    feature_corners[tcidl] = kdl;

    LandmarkMatchData md;
    find_matches_landmarks(kdl, active_landmarks, feature_corners,
                           projected_points, projected_track_ids,
                           match_max_dist_2d, feature_match_max_dist,
                           feature_match_test_next_best, md);

    std::cout << "Found " << md.matches.size() << " matches." << std::endl;

    // original:    localize_camera(calib_cam.intrinsics[0], kdl, landmarks,
    // reprojection_error_pnp_inlier_threshold_pixel, md);
    if (md.matches.size() >= 4) {
      localize_camera(calib_cam.intrinsics[0], kdl, landmarks,
                      reprojection_error_pnp_inlier_threshold_pixel, md);
    } else {
      md.T_w_c = current_pose;
    }

    current_pose = md.T_w_c;

    std::cout << "Found " << md.inliers.size() << " inliers." << std::endl;

    if (int(md.inliers.size()) < new_kf_min_inliers && !opt_running &&
        !opt_finished) {
      take_keyframe = true;
    }

    if (!opt_running && opt_finished) {
      opt_thread->join();
      // landmarks = landmarks_opt;
      // landmarks_opt only store the information about active landmarks
      for (const auto& lm : landmarks_opt) {
        if (landmarks.count(lm.first) != 0) {
          landmarks.find(lm.first)->second = lm.second;
        }
      }
      cameras = cameras_opt;
      // for (const auto& ca : cameras_opt) {
      //  cameras.find(ca.first)->second = ca.second;
      //}
      calib_cam = calib_cam_opt;

      opt_finished = false;
    }

    // update image views
    change_display_to_image(tcidl);
    change_display_to_image(tcidr);

    current_frame++;
    return true;
  }
}

// Compute reprojections for all landmark observations for visualization and
// outlier removal.
void compute_projections() {
  image_projections.clear();

  for (const auto& kv_lm : landmarks) {
    const TrackId track_id = kv_lm.first;

    for (const auto& kv_obs : kv_lm.second.obs) {
      const TimeCamId& tcid = kv_obs.first;
      const Eigen::Vector2d p_2d_corner =
          feature_corners.at(tcid).corners[kv_obs.second];

      const Eigen::Vector3d p_c =
          cameras.at(tcid).T_w_c.inverse() * kv_lm.second.p;
      const Eigen::Vector2d p_2d_repoj =
          calib_cam.intrinsics.at(tcid.cam_id)->project(p_c);

      ProjectedLandmarkPtr proj_lm(new ProjectedLandmark);
      proj_lm->track_id = track_id;
      proj_lm->point_measured = p_2d_corner;
      proj_lm->point_reprojected = p_2d_repoj;
      proj_lm->point_3d_c = p_c;
      proj_lm->reprojection_error = (p_2d_corner - p_2d_repoj).norm();

      image_projections[tcid].obs.push_back(proj_lm);
    }

    for (const auto& kv_obs : kv_lm.second.outlier_obs) {
      const TimeCamId& tcid = kv_obs.first;
      const Eigen::Vector2d p_2d_corner =
          feature_corners.at(tcid).corners[kv_obs.second];

      const Eigen::Vector3d p_c =
          cameras.at(tcid).T_w_c.inverse() * kv_lm.second.p;
      const Eigen::Vector2d p_2d_repoj =
          calib_cam.intrinsics.at(tcid.cam_id)->project(p_c);

      ProjectedLandmarkPtr proj_lm(new ProjectedLandmark);
      proj_lm->track_id = track_id;
      proj_lm->point_measured = p_2d_corner;
      proj_lm->point_reprojected = p_2d_repoj;
      proj_lm->point_3d_c = p_c;
      proj_lm->reprojection_error = (p_2d_corner - p_2d_repoj).norm();

      image_projections[tcid].outlier_obs.push_back(proj_lm);
    }
  }
}

// Optimize the active map with bundle adjustment
void optimize() {
  size_t num_obs = 0;
  for (const auto& kv : active_landmarks) {
    num_obs += kv.second.obs.size();
  }

  std::cerr << "Optimizing map with " << active_cameras.size() << ", "
            << active_landmarks.size() << " points and " << num_obs
            << " observations." << std::endl;

  // Fix oldest two cameras to fix SE3 and scale gauge. Making the whole
  // second camera constant is a bit suboptimal, since we only need 1 DoF, but
  // it's simple and the initial poses should be good from calibration.
  // FrameId fid = *(kf_frames.begin());
  // std::set<FrameId> fid;
  // std::cout << "fid " << fid << std::endl;

  std::set<TimeCamId> fixed_cameras_Id;
  for (const auto& fc : fixed_cameras) {
    fixed_cameras_Id.insert(fc.first);
  }

  // Prepare bundle adjustment
  BundleAdjustmentOptions ba_options;
  ba_options.optimize_intrinsics = ba_optimize_intrinsics;
  ba_options.use_huber = true;
  ba_options.huber_parameter = reprojection_error_huber_pixel;
  ba_options.max_num_iterations = 20;
  ba_options.verbosity_level = ba_verbose;

  calib_cam_opt = calib_cam;
  cameras_opt = cameras;
  // here is only the active landmarks
  landmarks_opt = active_landmarks;
  fixed_cameras_opt = fixed_cameras;
  active_cameras_opt = active_cameras;

  opt_running = true;

  // opt_thread.reset(new std::thread([fid, ba_options]
  opt_thread.reset(new std::thread([ba_options] {
    // std::set<TimeCamId> fixed_cameras = {{fid, 0}, {fid, 1}};
    local_bundle_adjustment(feature_corners, ba_options, calib_cam_opt,
                            fixed_cameras_opt, active_cameras_opt, cameras_opt,
                            landmarks_opt);

    opt_finished = true;
    opt_running = false;
  }));

  // Update project info cache
  compute_projections();
}
