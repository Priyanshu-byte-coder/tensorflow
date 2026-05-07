/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/backends/gpu/transforms/bitcast_transforms.h"

#include <cstdint>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

namespace xla::gpu {
namespace {

struct CommonFactorsTestCase {
  std::vector<int64_t> from, to;
  absl::InlinedVector<std::pair<int64_t, int64_t>, 8> expected;
};

class CommonFactorsMergingTrivialRangesTest
    : public ::testing::TestWithParam<CommonFactorsTestCase> {};

TEST_P(CommonFactorsMergingTrivialRangesTest, Example) {
  const CommonFactorsTestCase& test_case = GetParam();
  EXPECT_EQ(test_case.expected, detail::CommonFactorsMergingTrivialRanges(
                                    test_case.from, test_case.to));
}

INSTANTIATE_TEST_SUITE_P(
    CommonFactorsMergingTrivialRangesTestSuite,
    CommonFactorsMergingTrivialRangesTest,
    ::testing::Values(
        CommonFactorsTestCase{{1}, {}, {{0, 0}, {1, 0}}},
        CommonFactorsTestCase{{}, {1}, {{0, 0}, {0, 1}}},
        CommonFactorsTestCase{{}, {}, {{0, 0}}},
        CommonFactorsTestCase{{1, 2, 0}, {2, 0, 3}, {{0, 0}, {3, 3}}},
        CommonFactorsTestCase{{2, 3, 0}, {1, 0, 1000}, {{0, 0}, {3, 3}}},
        CommonFactorsTestCase{{1, 1, 1}, {1, 1}, {{0, 0}, {1, 1}, {3, 2}}},
        CommonFactorsTestCase{{1, 1, 3}, {3, 1, 1}, {{0, 0}, {3, 3}}},
        CommonFactorsTestCase{{2, 6}, {4, 3}, {{0, 0}, {2, 2}}},
        CommonFactorsTestCase{{1, 2, 6}, {4, 1, 3, 1}, {{0, 0}, {3, 4}}},
        CommonFactorsTestCase{{2, 3, 4, 5}, {6, 20}, {{0, 0}, {2, 1}, {4, 2}}},
        CommonFactorsTestCase{
            {2, 3, 4, 5, 6}, {6, 20, 6}, {{0, 0}, {2, 1}, {4, 2}, {5, 3}}},
        CommonFactorsTestCase{{2, 2, 2, 2}, {4, 4}, {{0, 0}, {2, 1}, {4, 2}}},
        CommonFactorsTestCase{
            {2, 5, 1, 3}, {1, 10, 3, 1}, {{0, 0}, {2, 2}, {4, 4}}}),
    [](const ::testing::TestParamInfo<CommonFactorsTestCase>& info) {
      return absl::StrCat(absl::StrJoin(info.param.from, "_"), "_to_",
                          absl::StrJoin(info.param.to, "_"));
    });

}  // namespace

}  // namespace xla::gpu
