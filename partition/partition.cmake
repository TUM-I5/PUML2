
option(USE_PARMETIS "Enable the ParMETIS (METIS) partitioning library." ON)
option(USE_PARHIP "Enable the ParHIP (KaHIP) partitioning library." ON)
option(USE_PTSCOTCH "Enable the PTSCOTCH (SCOTCH) partitioning library." ON)

if (USE_PARMETIS)
include(cmake/FindParMETIS)
endif()
if (USE_PARHIP)
include(cmake/FindParHIP)
endif()
if (USE_PTSCOTCH)
include(cmake/FindPTSCOTCH)
endif()
