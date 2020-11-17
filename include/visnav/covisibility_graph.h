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
#include <set>

namespace visnav {

void add_loop_closure_edge_covisibility_gragh(
    const FrameId& fid1, const FrameId& fid2,
    CovisibilityGraph& covisibility_graph, const int weight) {
  // if fid1 already exsits in essential graph
  if (covisibility_graph.count(fid1) != 0) {
    covisibility_graph.find(fid1)->second.insert(std::make_pair(fid2, weight));
  }
  // if fid1 does not exsit in essential graph
  else {
    std::map<FrameId, int> edge;
    edge.insert(std::make_pair(fid2, weight));
    covisibility_graph.insert(std::make_pair(fid1, edge));
  }
  // if fid2 already exsits in essential graph
  if (covisibility_graph.count(fid2) != 0) {
    covisibility_graph.find(fid2)->second.insert(std::make_pair(fid1, weight));
  }
  // if fid2 does not exsit in essential graph
  else {
    std::map<FrameId, int> edge;
    edge.insert(std::make_pair(fid1, weight));
    covisibility_graph.insert(std::make_pair(fid2, edge));
  }
}

void update_covisibility_graph(CovisibilityGraph& covisibility_gragh,
                               const TimeCamId& tcidl, const TimeCamId& tcidr,
                               const Cameras& cameras,
                               const Landmarks& landmarks,
                               const int num_covisibility_same_observations) {
  std::map<FrameId, int> edges;
  for (const auto& cam : cameras) {
    // don't count the same keyframe
    if (cam.first.t_ns == tcidl.t_ns) {
      continue;
    }
    if (cam.first.cam_id == tcidl.cam_id) {
      int weight_left = 0;
      int weight_right = 0;
      for (const auto& lm : landmarks) {
        // if they are the left camera
        if (lm.second.obs.count(cam.first) != 0 &&
            lm.second.obs.count(tcidl) != 0) {
          weight_left += 1;
        }

        // for right camera
        TimeCamId tidr(cam.first.t_ns, 1);

        if (lm.second.obs.count(tidr) != 0 && lm.second.obs.count(tcidr) != 0) {
          weight_right += 1;
        }
      }
      //  if one of two frames observe more than x same landmarks, the edge
      // between them will be added
      if (weight_left + weight_right >= num_covisibility_same_observations) {
        edges.insert(
            std::make_pair(cam.first.t_ns, weight_left + weight_right));
      }
    }
  }
  if (covisibility_gragh.count(tcidl.t_ns) == 0) {
    covisibility_gragh.insert(std::make_pair(tcidl.t_ns, edges));
  } else {
    covisibility_gragh.find(tcidl.t_ns)->second = edges;
  }
  // add this frame to its neigbors, too
  for (const auto& edge : edges) {
    if (covisibility_gragh.count(edge.first) != 0) {
      covisibility_gragh.find(edge.first)
          ->second.insert(std::make_pair(tcidl.t_ns, edge.second));
    } else {
      std::map<FrameId, int> edges_n;
      edges_n.insert(std::make_pair(tcidl.t_ns, edge.second));
      covisibility_gragh.insert(std::make_pair(edge.first, edges_n));
    }
  }

  // check if the neighbor is still in cameras
  for (auto& cg : covisibility_gragh) {
    for (auto it1 = cg.second.begin(); it1 != cg.second.end();) {
      TimeCamId tid(it1->first, 0);
      // if the neighbors is no more in the cameras
      if (cameras.count(tid) == 0) {
        cg.second.erase(it1++);
        if (cg.second.size() == 0) {
          break;
        }

      } else {
        ++it1;
      }
    }
  }

  // base on cameras update covisibility graph
  for (auto it = covisibility_gragh.begin(); it != covisibility_gragh.end();) {
    // it this frame doesn't exist in cameras anymore or if it has no edges
    TimeCamId tid(it->first, 0);
    if (cameras.count(tid) == 0 || it->second.size() == 0) {
      covisibility_gragh.erase(it++);
    } else {
      ++it;
    }
  }
}

// output every frame in covisibility graph and its neighbors
void print_covisibility_gragh(const CovisibilityGraph& covisibility_gragh) {
  for (const auto& cg : covisibility_gragh) {
    std::cout << "keyframe:" << cg.first << std::endl;
    for (const auto& nb : cg.second) {
      std::cout << "neighbor:" << nb.first << "with weigth" << nb.second
                << std::endl;
    }
  }
}

std::set<FrameId> return_neighbors_in_covisibility(
    const CovisibilityGraph& covisibility_gragh, const FrameId& fid,
    const int threshold) {
  std::set<FrameId> neighbors;
  std::map<FrameId, int> all_neighbors;
  if (covisibility_gragh.count(fid) != 0) {
    all_neighbors = covisibility_gragh.find(fid)->second;
    for (const auto& nb : all_neighbors) {
      if (nb.second >= threshold) {
        neighbors.insert(nb.first);
      }
    }
  }
  return neighbors;
}

std::set<std::pair<FrameId, int>> return_neighbors_in_covisibility_with_weight(
    const CovisibilityGraph& covisibility_gragh, const FrameId& fid,
    const int threshold) {
  std::set<std::pair<FrameId, int>> neighbors;
  std::map<FrameId, int> all_neighbors;
  if (covisibility_gragh.count(fid) != 0) {
    all_neighbors = covisibility_gragh.find(fid)->second;
    for (const auto& nb : all_neighbors) {
      if (nb.second >= threshold) {
        neighbors.insert(nb);
      }
    }
  }
  return neighbors;
}
}  // namespace visnav
