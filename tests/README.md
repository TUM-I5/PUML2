# PUML Test application

The test application reads a PUML mesh and writes the mesh using the [XdmfWriter](https://github.com/TUM-I5/XdmfWriter).
The application assumes that cell, vertices and cell data are stored and the same HDF5 file and have the names `connect`, `geometry` and `group`, respectively.

To compile the test application copy the [XdmfWriter](https://github.com/TUM-I5/XdmfWriter) into this directory.
