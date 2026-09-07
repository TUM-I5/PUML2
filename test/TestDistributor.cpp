// SPDX-FileCopyrightText: 2026 Technical University of Munich
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include "PUML.h"

namespace {

using PUML::Distributor;

/// Every entity belongs to exactly one rank, the ranks cover the entities in
/// order, and the local ids run from zero.
void checkCoverage(const Distributor& distributor, unsigned long entities, unsigned long ranks) {
  unsigned long seen = 0;
  unsigned long expectedOffset = 0;

  for (unsigned long rank = 0; rank < ranks; ++rank) {
    const auto [offset, size] = distributor.offsetAndSize(rank);
    EXPECT_EQ(offset, expectedOffset) << "rank " << rank;

    for (unsigned long i = 0; i < size; ++i) {
      const auto gid = offset + i;
      EXPECT_EQ(distributor.rankOfEntity(gid), rank) << "entity " << gid;
      EXPECT_EQ(distributor.globalToLocalId(rank, gid), i) << "entity " << gid;
    }

    seen += size;
    expectedOffset += size;
  }

  EXPECT_EQ(seen, entities);
  EXPECT_EQ(distributor.totalSize(), entities);
}

TEST(Distributor, EvenSplitCoversEveryEntity) {
  for (unsigned long entities = 0; entities <= 12; ++entities) {
    for (unsigned long ranks = 1; ranks <= 6; ++ranks) {
      checkCoverage(Distributor(entities, ranks), entities, ranks);
    }
  }
}

TEST(Distributor, EvenSplitIsBalanced) {
  for (unsigned long entities = 0; entities <= 12; ++entities) {
    for (unsigned long ranks = 1; ranks <= 6; ++ranks) {
      const Distributor distributor(entities, ranks);

      unsigned long smallest = entities;
      unsigned long largest = 0;
      for (unsigned long rank = 0; rank < ranks; ++rank) {
        const auto size = distributor.offsetAndSize(rank).second;
        smallest = std::min(smallest, size);
        largest = std::max(largest, size);
      }
      EXPECT_LE(largest - smallest, 1UL) << entities << " entities on " << ranks << " ranks";
    }
  }
}

TEST(Distributor, ExplicitSplitMatchesTheEvenSplit) {
  for (unsigned long entities = 0; entities <= 12; ++entities) {
    for (unsigned long ranks = 1; ranks <= 6; ++ranks) {
      const Distributor even(entities, ranks);

      std::vector<unsigned long> offsets(ranks + 1);
      for (unsigned long rank = 0; rank < ranks; ++rank) {
        offsets[rank + 1] = offsets[rank] + even.offsetAndSize(rank).second;
      }

      checkCoverage(Distributor(offsets), entities, ranks);
    }
  }
}

TEST(Distributor, ExplicitSplitHandlesEmptyRanks) {
  // Ranks 0 and 3 hold nothing.
  const Distributor distributor(std::vector<unsigned long>{0, 0, 3, 7, 7, 9});
  checkCoverage(distributor, 9, 5);

  EXPECT_EQ(distributor.offsetAndSize(0).second, 0UL);
  EXPECT_EQ(distributor.offsetAndSize(3).second, 0UL);
  EXPECT_EQ(distributor.rankOfEntity(0), 1UL);
  EXPECT_EQ(distributor.rankOfEntity(2), 1UL);
  EXPECT_EQ(distributor.rankOfEntity(3), 2UL);
  EXPECT_EQ(distributor.rankOfEntity(6), 2UL);
  EXPECT_EQ(distributor.rankOfEntity(7), 4UL);
}

TEST(Distributor, ExplicitSplitCanBeLopsided) {
  const Distributor distributor(std::vector<unsigned long>{0, 1, 2, 100});
  checkCoverage(distributor, 100, 3);
}

} // namespace
