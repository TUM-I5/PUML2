// SPDX-FileCopyrightText: 2026 Technical University of Munich
//
// SPDX-License-Identifier: BSD-3-Clause

// Every test runs on every rank, so the tests use EXPECT rather than ASSERT:
// leaving a test early on one rank only would deadlock the others in the next
// collective call.

#include <gtest/gtest.h>

#ifdef USE_MPI
#include <mpi.h>
#endif // USE_MPI

auto main(int argc, char** argv) -> int {
#ifdef USE_MPI
  MPI_Init(&argc, &argv);
#endif // USE_MPI

  ::testing::InitGoogleTest(&argc, argv);

  int rank = 0;
#ifdef USE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif // USE_MPI

  if (rank != 0) {
    auto& listeners = ::testing::UnitTest::GetInstance()->listeners();
    delete listeners.Release(listeners.default_result_printer());
  }

  int result = RUN_ALL_TESTS();

#ifdef USE_MPI
  MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Finalize();
#endif // USE_MPI

  return result;
}
