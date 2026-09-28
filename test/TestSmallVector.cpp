// SPDX-FileCopyrightText: 2026 Technical University of Munich
//
// SPDX-License-Identifier: BSD-3-Clause

#include <utility>

#include <gtest/gtest.h>

#include "SmallVector.h"
#include "Types.h"

namespace {

using PUML::internal::SmallVector;

/// Fills the values 100, 101, ... in, one push_back at a time.
auto countingUp(int count) -> SmallVector<int, 4> {
  SmallVector<int, 4> values;
  for (int i = 0; i < count; ++i) {
    values.push_back(100 + i);
  }
  return values;
}

void expectCountingUp(const SmallVector<int, 4>& values, int count) {
  ASSERT_EQ(values.size(), static_cast<PUML::Size>(count));
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(values[i], 100 + i) << "value " << i;
  }
}

/// Pushing past the values kept next to the object moves them to the heap,
/// more than once for this count, and every value comes along.
TEST(SmallVector, KeepsItsValuesWhenPushingPastItsCapacity) {
  expectCountingUp(countingUp(4), 4);
  expectCountingUp(countingUp(5), 5);
  expectCountingUp(countingUp(20), 20);
}

TEST(SmallVector, KeepsItsValuesWhenReservingMore) {
  auto values = countingUp(3);
  values.reserve(10);
  expectCountingUp(values, 3);

  values = countingUp(6);
  values.reserve(20);
  expectCountingUp(values, 6);
}

/// Growing by resize keeps the values there are and fills the new ones with zero.
TEST(SmallVector, KeepsItsValuesWhenResizingPastItsCapacity) {
  auto values = countingUp(3);
  values.resize(10);
  ASSERT_EQ(values.size(), 10U);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(values[i], 100 + i) << "value " << i;
  }
  for (int i = 3; i < 10; ++i) {
    EXPECT_EQ(values[i], 0) << "value " << i;
  }
}

TEST(SmallVector, CopiesAndMovesTheValuesOnTheHeap) {
  const auto values = countingUp(9);

  // the copy holds values of its own, which grow without touching the original
  auto copy = values;
  copy.push_back(109);
  expectCountingUp(copy, 10);
  expectCountingUp(values, 9);

  auto source = countingUp(9);
  const auto moved = std::move(source);
  expectCountingUp(moved, 9);

  auto small = countingUp(2);
  auto large = countingUp(9);
  small.swap(large);
  expectCountingUp(small, 9);
  expectCountingUp(large, 2);
}

} // namespace
