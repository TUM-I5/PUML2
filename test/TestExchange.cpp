// SPDX-FileCopyrightText: 2026 Technical University of Munich
//
// SPDX-License-Identifier: BSD-3-Clause

#include <array>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#ifdef USE_MPI
#include <mpi.h>
#endif // USE_MPI

#include "Error.h"
#include "Exchange.h"
#include "PumlTest.h"
#include "TypeInference.h"
#include "Types.h"

namespace {

using namespace puml::test;
using PUML::Size;
using PUML::internal::Exchange;

TEST(Exchange, LaysOutTheBlocksRankByRank) {
  const Exchange exchange({2, 0, 3}, {1, 4, 0});

  EXPECT_EQ(exchange.ranks(), 3);
  EXPECT_EQ(exchange.sendOffset(0), 0U);
  EXPECT_EQ(exchange.sendOffset(1), 2U);
  EXPECT_EQ(exchange.sendOffset(2), 2U);
  EXPECT_EQ(exchange.sendTotal(), 5U);
  EXPECT_EQ(exchange.recvOffset(1), 1U);
  EXPECT_EQ(exchange.recvOffset(2), 5U);
  EXPECT_EQ(exchange.recvTotal(), 5U);

  // what came in goes back out
  const auto reverse = exchange.reversed();
  EXPECT_EQ(reverse.sendCount(1), 4U);
  EXPECT_EQ(reverse.recvCount(2), 3U);
  EXPECT_EQ(reverse.sendOffset(2), 5U);
  EXPECT_EQ(reverse.recvOffset(1), 2U);

  const Exchange none;
  EXPECT_EQ(none.sendTotal(), 0U);
  EXPECT_EQ(none.recvTotal(), 0U);
}

#ifdef USE_MPI

/// How many entities a rank sends to another in the exchanges here: none to
/// some of them, and more than one to others.
auto countOf(int from, int to) -> Size { return static_cast<Size>((from + (2 * to)) % 4); }

/// What an entity says about where it came from and where it went to.
auto valueOf(int from, int to, Size index) -> PUML::GlobalId {
  return (static_cast<PUML::GlobalId>(from) * 1000000) + (static_cast<PUML::GlobalId>(to) * 1000) +
         index;
}

/// The entities a rank sends, of `width` values each: every block in turn.
auto sendBufferOf(const Exchange& exchange, int rank, Size width) -> std::vector<PUML::GlobalId> {
  std::vector<PUML::GlobalId> values(exchange.sendTotal() * width);
  for (int to = 0; to < exchange.ranks(); ++to) {
    for (Size i = 0; i < exchange.sendCount(to); ++i) {
      for (Size k = 0; k < width; ++k) {
        values[((exchange.sendOffset(to) + i) * width) + k] = valueOf(rank, to, (i * width) + k);
      }
    }
  }
  return values;
}

/// Whether a rank has received what every rank sent it, in the order of the ranks.
auto receivedAll(const Exchange& exchange,
                 const std::vector<PUML::GlobalId>& received,
                 int rank,
                 Size width) -> bool {
  bool all = received.size() == exchange.recvTotal() * width;
  for (int from = 0; all && from < exchange.ranks(); ++from) {
    for (Size i = 0; i < exchange.recvCount(from); ++i) {
      for (Size k = 0; k < width; ++k) {
        all = all && received[((exchange.recvOffset(from) + i) * width) + k] ==
                         valueOf(from, rank, (i * width) + k);
      }
    }
  }
  return all;
}

auto exchangeOfThisRank() -> Exchange {
  std::vector<Size> sendCounts(commSize());
  for (int to = 0; to < commSize(); ++to) {
    sendCounts[to] = countOf(commRank(), to);
  }
  return Exchange::ofSendCounts(sendCounts, MPI_COMM_WORLD);
}

TEST(Exchange, LearnsTheCountsFromTheOtherRanks) {
  const auto exchange = exchangeOfThisRank();
  for (int from = 0; from < commSize(); ++from) {
    EXPECT_EQ(exchange.recvCount(from), countOf(from, commRank())) << "from " << from;
  }
}

TEST(Exchange, MovesTheBlocksAllToAllAndPointToPoint) {
  const auto exchange = exchangeOfThisRank();

  // single values, and entities of three values under a type of their own
  MPI_Datatype triple = MPI_DATATYPE_NULL;
  MPI_Type_contiguous(3, PUML::MPITypeInfer<PUML::GlobalId>::type(), &triple);
  MPI_Type_commit(&triple);
  const std::array<std::pair<Size, MPI_Datatype>, 2> kinds = {
      {{1, PUML::MPITypeInfer<PUML::GlobalId>::type()}, {3, triple}}};

  for (const auto& [width, type] : kinds) {
    const auto sent = sendBufferOf(exchange, commRank(), width);

    std::vector<PUML::GlobalId> allToAll(exchange.recvTotal() * width);
    exchange.alltoallv(sent.data(), allToAll.data(), type, MPI_COMM_WORLD);
    EXPECT_TRUE(receivedAll(exchange, allToAll, commRank(), width)) << "width " << width;

    std::vector<PUML::GlobalId> pointToPoint(exchange.recvTotal() * width);
    exchange.pointToPoint(sent.data(), pointToPoint.data(), type, MPI_COMM_WORLD);
    EXPECT_EQ(pointToPoint, allToAll) << "width " << width;

    // and all of it back where it came from
    std::vector<PUML::GlobalId> returned(exchange.sendTotal() * width);
    exchange.reversed().alltoallv(allToAll.data(), returned.data(), type, MPI_COMM_WORLD);
    EXPECT_EQ(returned, sent) << "width " << width;
  }

  MPI_Type_free(&triple);
}

#ifndef PUML_MPI_LARGE_COUNT
/// Without the calls of MPI 4, an exchange that an int cannot count is refused
/// before anything is sent, on every rank alike, so that no buffer is touched.
TEST(Exchange, RefusesWhatAnIntCannotCount) {
  const auto tooMany = static_cast<Size>(std::numeric_limits<int>::max()) + 1;
  const auto messageOf = [](const Exchange& exchange, bool allToAll) -> std::string {
    try {
      if (allToAll) {
        exchange.alltoallv(nullptr, nullptr, MPI_BYTE, MPI_COMM_WORLD);
      } else {
        exchange.pointToPoint(nullptr, nullptr, MPI_BYTE, MPI_COMM_WORLD);
      }
    } catch (const PUML::Error& error) {
      return error.what();
    }
    return {};
  };

  // a single block too large
  std::vector<Size> counts(commSize());
  counts[0] = tooMany;
  const Exchange block(counts, counts);
  EXPECT_NE(messageOf(block, true).find("more than MPI can count"), std::string::npos);
  EXPECT_NE(messageOf(block, false).find("more than MPI can count"), std::string::npos);

  // blocks that fit an int on their own, but not where the last one starts
  if (commSize() >= 3) {
    const std::vector<Size> halves(commSize(), tooMany / 2);
    const Exchange blocks(halves, halves);
    EXPECT_NE(messageOf(blocks, true).find("more than MPI can count"), std::string::npos);
  }
}
#endif // PUML_MPI_LARGE_COUNT

#endif // USE_MPI

} // namespace
