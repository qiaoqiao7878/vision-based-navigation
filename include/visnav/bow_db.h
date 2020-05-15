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

#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_vector.h>

namespace visnav {

class BowDatabase {
 public:
  BowDatabase() {}

  inline void insert(const TimeCamId& tcid, const BowVector& bow_vector) {
    // TODO SHEET 3: add a bow_vector that corresponds to frame tcid to the
    // inverted index. You can assume the image hasn't been added before.

    for (size_t i = 0; i < bow_vector.size(); i++) {
      auto it = inverted_index.find(bow_vector[i].first);
      if (it == inverted_index.end()) {
        std::pair<TimeCamId, WordValue> pair;
        pair.first = tcid;
        pair.second = bow_vector[i].second;
        tbb::concurrent_vector<std::pair<TimeCamId, WordValue>> vec;
        vec.push_back(pair);
        WordId wordid = bow_vector[i].first;

        std::pair<WordId,
                  tbb::concurrent_vector<std::pair<TimeCamId, WordValue>>>
            pair2;
        pair2.first = wordid;
        pair2.second = vec;
        inverted_index.insert(pair2);
      } else {
        std::pair<TimeCamId, WordValue> pair;
        pair.first = tcid;
        pair.second = bow_vector[i].second;
        it->second.push_back(pair);
      }
    }

    UNUSED(tcid);
    UNUSED(bow_vector);
  }

  inline void query(const BowVector& bow_vector, size_t num_results,
                    BowQueryResult& results) const {
    // TODO SHEET 3: find num_results closest matches to the bow_vector in the
    // inverted index. Hint: for good query performance use std::unordered_map
    // to accumulate scores and std::partial_sort for getting the closest
    // results. You should use L1 difference as the distance measure. You can
    // assume that BoW descripors are L1 normalized.

    // Normalized sparse vector of words to represent images. "Sparse" means
    // that words with value 0 don't appear explicitly.
    // using BowVector = std::vector<std::pair<WordId, WordValue>>;

    // Result of BoW query. Should be sorted by the confidence.
    // using BowQueryResult = std::vector<std::pair<TimeCamId, double>>;
    // std::cout << "bow: " << bow_vector.size() << "\n"; 978
    // std::cout << "num: " << num_results << "\n"; 20
    std::unordered_map<TimeCamId, double> score;
    for (size_t i = 0; i < bow_vector.size(); i++) {
      auto it = inverted_index.find(bow_vector[i].first);
      if (it != inverted_index.end()) {
        tbb::concurrent_vector<std::pair<TimeCamId, WordValue>> vec;
        vec = it->second;
        for (size_t j = 0; j < vec.size(); j++) {
          auto it_1 = score.find(vec[j].first);
          if (it_1 == score.end()) {
            std::pair<TimeCamId, double> pair;

            pair.first = vec[j].first;
            pair.second = 2 + abs(vec[j].second - bow_vector[i].second) -
                          abs(vec[j].second) - std::abs(bow_vector[i].second);
            score.insert(pair);
          } else {
            score.find(vec[j].first)->second +=
                abs(vec[j].second - bow_vector[i].second) - abs(vec[j].second) -
                std::abs(bow_vector[i].second);
          }
        }
      }
    }

    // Get an iterator pointing to begining of map
    auto it_2 = score.begin();

    // Iterate over the map using iterator
    while (it_2 != score.end()) {
      std::pair<TimeCamId, WordValue> pair;
      pair.first = it_2->first;
      pair.second = it_2->second;
      results.push_back(pair);
      it_2++;
    }
    std::partial_sort(
        results.begin(), results.begin() + num_results, results.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });
    results.resize(num_results);
    UNUSED(bow_vector);
    UNUSED(num_results);
    UNUSED(results);
  }

  void clear() { inverted_index.clear(); }

 protected:
  tbb::concurrent_unordered_map<
      WordId, tbb::concurrent_vector<std::pair<TimeCamId, WordValue>>>
      inverted_index;
};

}  // namespace visnav
