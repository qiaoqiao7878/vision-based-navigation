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

#include <visnav/covisibility_graph.h>

namespace visnav {

void add_loop_closure_edge_essential_gragh(const FrameId& fid1,
                                           const FrameId& fid2,
                                           EssentialGraph& essential_graph,
                                           const int weight) {
  // if fid1 already exsits in essential graph
  if (essential_graph.count(fid1) != 0) {
    essential_graph.find(fid1)->second.insert(std::make_pair(fid2, weight));
  }
  // if fid1 does not exsit in essential graph
  else {
    std::map<FrameId, int> edge;
    edge.insert(std::make_pair(fid2, weight));
    essential_graph.insert(std::make_pair(fid1, edge));
  }
  // if fid2 already exsits in essential graph
  if (essential_graph.count(fid2) != 0) {
    essential_graph.find(fid2)->second.insert(std::make_pair(fid1, weight));
  }
  // if fid2 does not exsit in essential graph
  else {
    std::map<FrameId, int> edge;
    edge.insert(std::make_pair(fid1, weight));
    essential_graph.insert(std::make_pair(fid2, edge));
  }
}

void add_essential_gragh_edges(const CovisibilityGraph& covisibility_gragh,
                               const FrameId& fid,
                               EssentialGraph& essential_graph,
                               const int num_covisibility_same_observations,
                               const int high_covisibility) {
  std::map<FrameId, int> edges;
  std::set<std::pair<FrameId, int>> neighbors;
  // get the neighbors from covisibility gragh
  neighbors = return_neighbors_in_covisibility_with_weight(
      covisibility_gragh, fid, num_covisibility_same_observations);

  if (neighbors.size() != 0) {
    FrameId most_common_neighbor = 0;
    int most_common = 0;
    for (const auto& nb : neighbors) {
      // add the edges with high covisibility
      if (nb.second >= high_covisibility) {
        edges.insert(nb);
      }
      // store the neighbor that share most common observations
      if (nb.second >= most_common) {
        most_common = nb.second;
        most_common_neighbor = nb.first;
      }
    }
    // if the edges with highest observation number under high covisibility
    // add this edge, too
    if (most_common < high_covisibility) {
      std::pair<FrameId, int> essential_neighbor;
      essential_neighbor.first = most_common_neighbor;
      essential_neighbor.second = most_common;
      edges.insert(essential_neighbor);
    }
  }
  if (essential_graph.count(fid) == 0) {
    essential_graph.insert(std::make_pair(fid, edges));
  } else {
    if (edges.size() != 0) {
      essential_graph.find(fid)->second = edges;
    }
  }

  // add this frame to its neigbors, too
  for (const auto& edge : edges) {
    if (essential_graph.count(edge.first) != 0) {
      essential_graph.find(edge.first)
          ->second.insert(std::make_pair(fid, edge.second));
    } else {
      std::map<FrameId, int> edges_n;
      edges_n.insert(std::make_pair(fid, edge.second));
      essential_graph.insert(std::make_pair(edge.first, edges_n));
    }
  }
}

void update_essential_graph(CovisibilityGraph& covisibility_gragh,
                            const FrameId& fid, const Cameras& cameras,
                            EssentialGraph& essential_graph,
                            const int num_covisibility_same_observations,
                            const int high_covisibility) {
  // if this FrameId doesn't exsit in essential graph

  add_essential_gragh_edges(covisibility_gragh, fid, essential_graph,
                            num_covisibility_same_observations,
                            high_covisibility);

  // check if the neighbor is still in cameras
  for (auto& cg : essential_graph) {
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

  // check if the node is still in cameras
  for (auto it = essential_graph.begin(); it != essential_graph.end();) {
    TimeCamId tid(it->first, 0);
    if (cameras.count(tid) == 0) {
      essential_graph.erase(it++);
    } else {
      ++it;
    }
  }
  for (auto it = essential_graph.begin(); it != essential_graph.end();) {
    TimeCamId tid(it->first, 0);
    if (it->second.size() == 0) {
      FrameId fid = it->first;
      essential_graph.erase(it++);
      add_essential_gragh_edges(covisibility_gragh, fid, essential_graph,
                                num_covisibility_same_observations,
                                high_covisibility);
    } else {
      ++it;
    }
  }
}

void print_essential_gragh(const EssentialGraph& essential_graph) {
  for (const auto& eg : essential_graph) {
    std::cout << "keyframe:" << eg.first << std::endl;
    for (const auto& nb : eg.second) {
      std::cout << "neighbor linked in essential graph::" << nb.first
                << "with weigth" << nb.second << std::endl;
    }
  }
}

}  // namespace visnav
