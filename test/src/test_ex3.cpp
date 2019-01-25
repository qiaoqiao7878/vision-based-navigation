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

#include <gtest/gtest.h>

#include <pangolin/image/image.h>
#include <pangolin/image/image_io.h>
#include <pangolin/image/typed_image.h>

#include "visnav/keypoints.h"
#include "visnav/matching_utils.h"

#include "visnav/serialization.h"

#include "visnav/bow_db.h"
#include "visnav/bow_voc.h"

#include <fstream>
#include <random>

using namespace visnav;

const int NUM_FEATURES = 1500;
const int MATCH_THRESHOLD = 70;
const double DIST_2_BEST = 1.2;

const std::string img0_path = "../../test/ex3_test_data/0_0.jpg";
const std::string img1_path = "../../test/ex3_test_data/0_1.jpg";

const std::string kd0_path = "../../test/ex3_test_data/kd0.json";
const std::string kd1_path = "../../test/ex3_test_data/kd1.json";

const std::string matches_stereo_path =
    "../../test/ex3_test_data/matches_stereo.json";
const std::string matches_path = "../../test/ex3_test_data/matches.json";

const std::string calib_path = "../../test/ex3_test_data/calib.json";

const std::string vocab_path = "../../data/ORBvoc.cereal";
const std::string bow_res_path = "../../test/ex3_test_data/bow_res.json";
const std::string bow_res2_path = "../../test/ex3_test_data/bow_res2.json";

TEST(Ex3TestSuite, KeypointAngles) {
  pangolin::ManagedImage<uint8_t> img0 = pangolin::LoadImage(img0_path);
  pangolin::ManagedImage<uint8_t> img1 = pangolin::LoadImage(img1_path);

  KeypointsData kd0, kd1, kd0_loaded, kd1_loaded;

  detectKeypointsAndDescriptors(img0, kd0, NUM_FEATURES, true);
  detectKeypointsAndDescriptors(img1, kd1, NUM_FEATURES, true);

  {
    std::ifstream os(kd0_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kd0_loaded);
  }

  {
    std::ifstream os(kd1_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kd1_loaded);
  }

  ASSERT_TRUE(kd0_loaded.corner_angles.size() == kd0.corner_angles.size());
  ASSERT_TRUE(kd1_loaded.corner_angles.size() == kd1.corner_angles.size());

  for (size_t i = 0; i < kd0_loaded.corner_angles.size(); i++) {
    ASSERT_TRUE(std::abs(kd0_loaded.corner_angles[i] - kd0.corner_angles[i]) <
                1e-8);
  }

  for (size_t i = 0; i < kd1_loaded.corner_angles.size(); i++) {
    ASSERT_TRUE(std::abs(kd1_loaded.corner_angles[i] - kd1.corner_angles[i]) <
                1e-8);
  }
}

TEST(Ex3TestSuite, KeypointDescriptors) {
  pangolin::ManagedImage<uint8_t> img0 = pangolin::LoadImage(img0_path);
  pangolin::ManagedImage<uint8_t> img1 = pangolin::LoadImage(img1_path);

  KeypointsData kd0, kd1, kd0_loaded, kd1_loaded;

  detectKeypointsAndDescriptors(img0, kd0, NUM_FEATURES, true);
  detectKeypointsAndDescriptors(img1, kd1, NUM_FEATURES, true);

  {
    std::ifstream os(kd0_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kd0_loaded);
  }

  {
    std::ifstream os(kd1_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kd1_loaded);
  }

  ASSERT_TRUE(kd0_loaded.corner_descriptors.size() ==
              kd0.corner_descriptors.size());
  ASSERT_TRUE(kd1_loaded.corner_descriptors.size() ==
              kd1.corner_descriptors.size());

  for (size_t i = 0; i < kd0_loaded.corner_descriptors.size(); i++) {
    ASSERT_TRUE((kd0_loaded.corner_descriptors[i] ^ kd0.corner_descriptors[i])
                    .count() == 0);
  }

  for (size_t i = 0; i < kd1_loaded.corner_descriptors.size(); i++) {
    ASSERT_TRUE((kd1_loaded.corner_descriptors[i] ^ kd1.corner_descriptors[i])
                    .count() == 0);
  }
}

TEST(Ex3TestSuite, DescriptorMatching) {
  MatchData md, md_loaded;
  KeypointsData kd0_loaded, kd1_loaded;

  {
    std::ifstream os(kd0_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kd0_loaded);
  }

  {
    std::ifstream os(kd1_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kd1_loaded);
  }

  matchDescriptors(kd0_loaded.corner_descriptors, kd1_loaded.corner_descriptors,
                   md.matches, MATCH_THRESHOLD, DIST_2_BEST);

  {
    std::ifstream os(matches_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(md_loaded);
  }

  ASSERT_TRUE(md_loaded.matches.size() == md.matches.size())
      << "md_loaded.matches.size() " << md_loaded.matches.size()
      << " md.matches.size() " << md.matches.size();

  for (size_t i = 0; i < md_loaded.matches.size(); i++) {
    ASSERT_TRUE(md_loaded.matches[i] == md.matches[i]);
  }
}

TEST(Ex3TestSuite, KeypointsAll) {
  pangolin::ManagedImage<uint8_t> img0 = pangolin::LoadImage(img0_path);
  pangolin::ManagedImage<uint8_t> img1 = pangolin::LoadImage(img1_path);

  MatchData md, md_loaded;
  KeypointsData kd0, kd1, kd0_loaded, kd1_loaded;

  detectKeypointsAndDescriptors(img0, kd0, NUM_FEATURES, true);
  detectKeypointsAndDescriptors(img1, kd1, NUM_FEATURES, true);

  matchDescriptors(kd0.corner_descriptors, kd1.corner_descriptors, md.matches,
                   MATCH_THRESHOLD, DIST_2_BEST);

  {
    std::ifstream os(kd0_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kd0_loaded);
  }

  {
    std::ifstream os(kd1_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kd1_loaded);
  }

  {
    std::ifstream os(matches_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(md_loaded);
  }

  ASSERT_TRUE(md_loaded.matches.size() == md.matches.size())
      << "md_loaded.matches.size() " << md_loaded.matches.size()
      << " md.matches.size() " << md.matches.size();

  for (size_t i = 0; i < md_loaded.matches.size(); i++) {
    ASSERT_TRUE(md_loaded.matches[i] == md.matches[i]);
  }
}

TEST(Ex3TestSuite, EpipolarInliers) {
  Calibration calib;

  MatchData md, md_loaded;
  KeypointsData kd0_loaded, kd1_loaded;

  {
    std::ifstream os(calib_path, std::ios::binary);

    if (os.is_open()) {
      cereal::JSONInputArchive archive(os);
      archive(calib);
    } else {
      ASSERT_TRUE(false) << "could not load camera ";
    }
  }

  {
    std::ifstream os(kd0_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kd0_loaded);
  }

  {
    std::ifstream os(kd1_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kd1_loaded);
  }

  {
    std::ifstream os(matches_stereo_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(md_loaded);
  }

  // Essential matrix
  Eigen::Matrix3d E;
  Sophus::SE3d T_0_1 = calib.T_i_c[0].inverse() * calib.T_i_c[1];

  computeEssential(T_0_1, E);

  md.matches = md_loaded.matches;
  findInliersEssential(kd0_loaded, kd1_loaded, calib.intrinsics[0],
                       calib.intrinsics[1], E, 1e-3, md);

  ASSERT_TRUE(md_loaded.inliers.size() == md.inliers.size())
      << "md_loaded.inliers.size() " << md_loaded.inliers.size()
      << " md.inliers.size() " << md.inliers.size();

  for (size_t i = 0; i < md_loaded.inliers.size(); i++) {
    ASSERT_TRUE(md_loaded.inliers[i] == md.inliers[i]);
  }
}

TEST(Ex3TestSuite, RansacInliers) {
  Calibration calib;

  MatchData md, md_loaded;
  KeypointsData kd0_loaded, kd1_loaded;

  {
    std::ifstream os(calib_path, std::ios::binary);

    if (os.is_open()) {
      cereal::JSONInputArchive archive(os);
      archive(calib);
    } else {
      ASSERT_TRUE(false) << "could not load camera ";
    }
  }

  {
    std::ifstream os(kd0_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kd0_loaded);
  }

  {
    std::ifstream os(kd1_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kd1_loaded);
  }

  {
    std::ifstream os(matches_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(md_loaded);
  }

  md.matches = md_loaded.matches;
  findInliersRansac(kd0_loaded, kd1_loaded, calib.intrinsics[0],
                    calib.intrinsics[1], 1e-5, 20, md);

  // Translation is only determined up to scale, so normalize before comparison.
  const double dist = (md_loaded.T_i_j.translation().normalized() -
                       md.T_i_j.translation().normalized())
                          .norm();
  const double angle = md_loaded.T_i_j.unit_quaternion().angularDistance(
      md.T_i_j.unit_quaternion());

  const int inlier_count_diff =
      std::abs(int(md_loaded.inliers.size()) - int(md.inliers.size()));

  std::set<std::pair<int, int>> md_loaded_inliers(md_loaded.inliers.begin(),
                                                  md_loaded.inliers.end()),
      md_inliers(md.inliers.begin(), md.inliers.end()), md_itersection_inliers;

  // compute set intersection
  size_t max_size = std::max(md_loaded_inliers.size(), md_inliers.size());

  std::set_intersection(
      md_loaded_inliers.begin(), md_loaded_inliers.end(), md_inliers.begin(),
      md_inliers.end(),
      std::inserter(md_itersection_inliers, md_itersection_inliers.begin()));

  double intersection_fraction =
      double(md_itersection_inliers.size()) / max_size;
  ASSERT_TRUE(intersection_fraction > 0.99)
      << "intersection_fraction " << intersection_fraction;

  ASSERT_TRUE(inlier_count_diff < 20) << "inlier " << inlier_count_diff;
  ASSERT_TRUE(dist < 0.05) << "dist " << dist;
  ASSERT_TRUE(angle < 0.01) << "angle " << angle;
}

TEST(Ex3TestSuite, BowSingleFeatureTransform) {
  KeypointsData kd0;
  std::vector<WordId> w_id, w_id_loaded;
  std::vector<WordValue> w_val, w_val_loaded;

  {
    std::ifstream os(kd0_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kd0);
  }

  {
    std::ifstream os(bow_res_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(cereal::make_nvp("w_id", w_id_loaded));
    archive(cereal::make_nvp("w_val", w_val_loaded));
  }

  BowVocabulary voc(vocab_path);

  w_id.resize(kd0.corner_descriptors.size());
  w_val.resize(kd0.corner_descriptors.size());

  for (size_t i = 0; i < kd0.corner_descriptors.size(); i++) {
    voc.transformFeatureToWord(kd0.corner_descriptors[i], w_id[i], w_val[i]);
  }

  ASSERT_TRUE(w_id.size() == w_id_loaded.size());
  ASSERT_TRUE(w_val.size() == w_val_loaded.size());

  for (size_t i = 0; i < kd0.corner_descriptors.size(); i++) {
    ASSERT_TRUE(w_id[i] == w_id_loaded[i]);
    ASSERT_TRUE(std::abs(w_val[i] - w_val_loaded[i]) < 1e-10);
  }
}

TEST(Ex3TestSuite, BowVectorTransform) {
  KeypointsData kd0, kd1;

  BowVector bow0, bow1, bow0_loaded, bow1_loaded;

  {
    std::ifstream os(kd0_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kd0);
  }

  {
    std::ifstream os(kd1_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kd1);
  }

  {
    std::ifstream os(bow_res2_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(cereal::make_nvp("bow0", bow0_loaded));
    archive(cereal::make_nvp("bow1", bow1_loaded));
  }

  BowVocabulary voc(vocab_path);
  voc.transform(kd0.corner_descriptors, bow0);
  voc.transform(kd1.corner_descriptors, bow1);

  {
    std::unordered_map<WordId, WordValue> m(bow0_loaded.begin(),
                                            bow0_loaded.end());
    double sum = 0;
    for (const auto& kv : bow0) {
      sum += std::abs(kv.second);

      auto it = m.find(kv.first);
      ASSERT_TRUE(it != m.end());
      ASSERT_TRUE(std::abs(it->second - kv.second) < 1e-10);
    }

    ASSERT_TRUE(std::abs(sum - 1.0) < 1e-10) << sum;
  }

  {
    std::unordered_map<WordId, WordValue> m(bow1_loaded.begin(),
                                            bow1_loaded.end());

    double sum = 0;
    for (const auto& kv : bow1) {
      sum += std::abs(kv.second);

      auto it = m.find(kv.first);
      ASSERT_TRUE(it != m.end());
      ASSERT_TRUE(std::abs(it->second - kv.second) < 1e-10);
    }

    ASSERT_TRUE(std::abs(sum - 1.0) < 1e-10) << sum;
  }
}

TEST(Ex3TestSuite, BowDB) {
  constexpr int Dsize = 256;
  constexpr int N = 100;
  constexpr int NF = 1000;
  constexpr int num_bow_candidates = 20;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::bernoulli_distribution d(0.5);

  auto rand_desc = [&]() {
    std::bitset<Dsize> bits;
    for (int n = 0; n < Dsize; ++n) {
      bits[n] = d(gen);
    }
    return bits;
  };

  auto bow_dist = [](const BowVector& b0, const BowVector& b1) {
    std::unordered_map<WordId, WordValue> m(b1.begin(), b1.end());

    double dist = 2.0;
    for (const auto& kv : b0) {
      auto it = m.find(kv.first);
      if (it != m.end()) {
        dist += std::abs(kv.second - it->second) - std::abs(kv.second) -
                std::abs(it->second);
      }
    }

    return dist;
  };

  BowVocabulary voc(vocab_path);
  BowDatabase bow_db;

  std::vector<BowVector> bows(N);

  for (int j = 0; j < N; j++) {
    std::vector<std::bitset<Dsize>> v;
    v.resize(NF);
    for (int i = 0; i < NF; i++) {
      v[i] = rand_desc();
    }
    voc.transform(v, bows[j]);

    TimeCamId tcid(j, 0);
    bow_db.insert(tcid, bows[j]);
  }

  // Matrix of distances between BoWs
  Eigen::MatrixXd m;
  m.setZero(N, N);

  for (int i = 0; i < N; i++) {
    for (int j = i + 1; j < N; j++) {
      m(i, j) = m(j, i) = bow_dist(bows[i], bows[j]);
    }
  }

  for (int j = 0; j < N; j++) {
    BowQueryResult res;
    bow_db.query(bows[j], num_bow_candidates, res);

    std::set<int> db_res;
    for (const auto& kv : res) {
      db_res.insert(kv.first.t_ns);
    }
    std::cerr << std::endl;

    std::vector<std::pair<int, double>> col_val;
    for (int i = 0; i < N; i++) {
      col_val.emplace_back(i, m(j, i));
    }

    std::partial_sort(
        col_val.begin(), col_val.begin() + num_bow_candidates, col_val.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });
    col_val.resize(num_bow_candidates);

    std::set<int> bf_res;
    for (const auto& kv : col_val) {
      bf_res.insert(kv.first);
    }
    std::cerr << std::endl;

    // Check that closest bows found by the database and brute force are the
    // same
    ASSERT_TRUE(db_res == bf_res);
  }
}
