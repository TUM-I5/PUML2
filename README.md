<!--
    SPDX-FileCopyrightText: 2017-2024 Technical University of Munich

    SPDX-License-Identifier: BSD-3-Clause
-->

# PUML2
A **P**arallel **U**nstructured **M**esh **L**ibrary optimized for reading large unstructured meshes, version **2**.

PUML2 is a parallel reader for large unstructured meshes.
Currently, it supports meshes stored in [HDF5](https://support.hdfgroup.org/HDF5/), described by the [XDMF format](http://xdmf.org).
PUML2 reads meshes in $\mathcal{O}(n + m/n)$ and requires $\mathcal{O}(n + m/n)$ memory per MPI process where $n$ is the total number of processes and $m$ the number of elements.
The current implementation is limited to tetrahedral meshes but the code is prepared to support hexahedral meshes as well.

PUML2 is a C++ header-only library written in C++17.
You simply copy the header files into your project and start using it.
