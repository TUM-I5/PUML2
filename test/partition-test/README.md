To run the partitioning test, do the following:

* create a subdirectory, e.g. `build`, right in this folder and run CMake there. You will in particular need ParMETIS, ParHIP and PTSCOTCH.
* run the resulting executable `tests` with the following arguments: `./tests ../setup.json ../weighting.json ../sample.json`, and wait until completion.

(NOTE: PTSCOTCH does right now not work well with edge weights, therefore these are explicitly ignored in `sample.json`)
