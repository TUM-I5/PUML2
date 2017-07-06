# PUML2
a Parallel Unstructured Mesh Library for reading large unstructured meshes

PUML2 is a parallel reader for large unstructured meshes.
Currently, it supports meshes stored in the [XDMF format](http://xdmf.org) with [HDF5](https://support.hdfgroup.org/HDF5/).
PUML2 reads meshes in O(n + m/n) and requires O(n + m/n) memory per MPI process where n is the total number of processes and m the number of elements.
The current implementation is limited to tetrahedral meshes but the code is prepared to support hexahedral meshes as well.

PUML2 is a C++ header-only library based on C++11.
You simply copy the header files into your project and start using it.
