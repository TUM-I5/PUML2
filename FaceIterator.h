/**
 * @file
 *  This file is part of PUML
 *
 *  For conditions of distribution and use, please see the copyright
 *  notice in the file 'COPYING' at the root directory of this package
 *  and the copyright notice at https://github.com/TUM-I5/PUMGen
 *
 * @copyright 2019 Technische Universitaet Muenchen
 * @author David Schneller <david.schneller@tum.de>
 */

#ifndef PUML_FACE_ITERATOR_H
#define PUML_FACE_ITERATOR_H

#ifdef USE_MPI
#include <mpi.h>
#endif // USE_MPI

#include <algorithm>
#include <vector>
#include <unordered_map>
#include <cassert>
#include <cstddef>
#include <typeinfo>
#include "Upward.h"

#include "utils/logger.h"

namespace PUML
{

#ifdef USE_MPI
template<typename T> class MPITypeInfer;
template<> class MPITypeInfer<char> { public: static MPI_Datatype type() { return MPI_CHAR; } };
template<> class MPITypeInfer<signed char> { public: static MPI_Datatype type() { return MPI_SIGNED_CHAR; } };
template<> class MPITypeInfer<unsigned char> { public: static MPI_Datatype type() { return MPI_UNSIGNED_CHAR; } };
template<> class MPITypeInfer<short> { public: static MPI_Datatype type() { return MPI_SHORT; } };
template<> class MPITypeInfer<unsigned short> { public: static MPI_Datatype type() { return MPI_UNSIGNED_SHORT; } };
template<> class MPITypeInfer<int> { public: static MPI_Datatype type() { return MPI_INT; } };
template<> class MPITypeInfer<unsigned> { public: static MPI_Datatype type() { return MPI_UNSIGNED; } };
template<> class MPITypeInfer<long> { public: static MPI_Datatype type() { return MPI_LONG; } };
template<> class MPITypeInfer<unsigned long> { public: static MPI_Datatype type() { return MPI_UNSIGNED_LONG; } };
template<> class MPITypeInfer<long long> { public: static MPI_Datatype type() { return MPI_LONG_LONG; } };
template<> class MPITypeInfer<unsigned long long> { public: static MPI_Datatype type() { return MPI_UNSIGNED_LONG_LONG; } };
template<> class MPITypeInfer<float> { public: static MPI_Datatype type() { return MPI_FLOAT; } };
template<> class MPITypeInfer<double> { public: static MPI_Datatype type() { return MPI_DOUBLE; } };
template<> class MPITypeInfer<long double> { public: static MPI_Datatype type() { return MPI_LONG_DOUBLE; } };
template<> class MPITypeInfer<wchar_t> { public: static MPI_Datatype type() { return MPI_WCHAR; } };
#endif

template<TopoType Topo>
class FaceIterator
{
private:
    class Face;
public:
    FaceIterator(const PUML<Topo>& puml, bool sparseComm = true) : m_sparseComm(sparseComm), m_puml(puml)
    {
        int rank = 0, commSize = 1;

#ifdef USE_MPI
        MPI_Comm_rank(m_puml.comm(), &rank);
        MPI_Comm_size(m_puml.comm(), &commSize);

        const auto& faces = puml.faces();

        std::vector<std::vector<int>> transfer(commSize);
        for (int i = 0; i < faces.size(); ++i) {
            int lid[2];
            Upward::cells(puml, faces[i], lid);
            assert(!(lid[0] == -1 && lid[1] != -1));
            assert(faces[i].shared().size() <= 1);
            if (lid[0] != -1 && faces[i].isShared()) {
                transfer[faces[i].shared()[0]].push_back(i);
            }
        }

        m_transferSize = std::vector<int>(commSize);
        m_transferDisp = std::vector<int>(commSize+1);
        m_transferDisp[0] = 0;
        for (int i = 0; i < commSize; ++i) {
            m_transferSize[i] = transfer[i].size();
            m_transferDisp[i+1] = m_transferDisp[i]+m_transferSize[i];
            std::sort(transfer[i].begin(), transfer[i].end(),
                [&faces] (int a, int b) -> bool {
                    return faces[a].gid() < faces[b].gid();
                });
        }

        m_transferCell = std::vector<int>(m_transferDisp[commSize]);
        m_transferFace = std::vector<int>(m_transferDisp[commSize]);
        for (int i = 0, j = 0; i < commSize; ++i) {
            for (int k = 0; k < m_transferSize[i]; ++j, ++k) {
                const auto& face = faces[transfer[i][k]];
                int lid[2];
                Upward::cells(puml, face, lid);
                m_transferCell[j] = lid[0];
                m_transferFace[j] = transfer[i][k];
            }
        }
#endif
    }
    
    template<typename T>
    void iterate(const std::vector<T>& cellData,
                        const std::function<void(int,int,const T&,const T&)>& faceHandler,
                        const std::function<void(int,int)>& boundaryFaceHandler = [](int a, int b){},
                        MPI_Datatype mpit = MPITypeInfer<T>::type()) {
        const auto& cellHandler = [&cellData](int fid, int cid){return cellData[cid];};
        iterate(cellHandler, faceHandler, boundaryFaceHandler, mpit);
    }

    template<typename T, typename S>
    void iterate(const std::vector<T>& external_cellData,
                const std::vector<S>& internal_cellData,
                        const std::function<void(int,int,const T&,const S&)>& faceHandler,
                        const std::function<void(int,int)>& boundaryFaceHandler = [](int a, int b){},
                        MPI_Datatype mpit = MPITypeInfer<T>::type()) {
        const auto& externalCellHandler = [&external_cellData](int fid, int cid){return external_cellData[cid];};
        const auto& internalCellHandler = [&internal_cellData](int fid, int cid){return internal_cellData[cid];};
        iterate(externalCellHandler, internalCellHandler, faceHandler, boundaryFaceHandler, mpit);
    }

    template<typename T>
    void iterate(const std::vector<T>& external_cellData,
                        const std::function<void(int,int,const T&)>& faceHandler,
                        const std::function<void(int,int)>& boundaryFaceHandler = [](int a, int b){},
                        MPI_Datatype mpit = MPITypeInfer<T>::type()) {
        const auto& externalCellHandler = [&external_cellData](int fid, int cid){return external_cellData[cid];};
        iterate_internal(externalCellHandler, faceHandler, boundaryFaceHandler, mpit);
    }

    template<typename T>
    void iterate(const T* cellData,
                        const std::function<void(int,int,const T&,const T&)>& faceHandler,
                        const std::function<void(int,int)>& boundaryFaceHandler = [](int a, int b){},
                        MPI_Datatype mpit = MPITypeInfer<T>::type()) {
        const auto& cellHandler = [&cellData](int fid, int cid){return cellData[cid];};
        iterate(cellHandler, faceHandler, boundaryFaceHandler, mpit);
    }

    template<typename T, typename S>
    void iterate(const T* external_cellData,
                const S* internal_cellData,
                        const std::function<void(int,int,const T&,const S&)>& faceHandler,
                        const std::function<void(int,int)>& boundaryFaceHandler = [](int a, int b){},
                        MPI_Datatype mpit = MPITypeInfer<T>::type()) {
        const auto& externalCellHandler = [&external_cellData](int fid, int cid){return external_cellData[cid];};
        const auto& internalCellHandler = [&internal_cellData](int fid, int cid){return internal_cellData[cid];};
        iterate(externalCellHandler, internalCellHandler, faceHandler, boundaryFaceHandler, mpit);
    }

    template<typename T>
    void iterate(const T* external_cellData,
                        const std::function<void(int,int,const T&)>& faceHandler,
                        const std::function<void(int,int)>& boundaryFaceHandler = [](int a, int b){},
                        MPI_Datatype mpit = MPITypeInfer<T>::type()) {
        const auto& externalCellHandler = [&external_cellData](int fid, int cid){return external_cellData[cid];};
        iterate_internal(externalCellHandler, faceHandler, boundaryFaceHandler, mpit);
    }

    template<typename T>
    void iterate(const std::function<T(int,int)>& cellHandler,
                        const std::function<void(int,int,const T&,const T&)>& faceHandler,
                        const std::function<void(int,int)>& boundaryFaceHandler = [](int a, int b){},
                        MPI_Datatype mpit = MPITypeInfer<T>::type()) {
        iterate(cellHandler, cellHandler, faceHandler, boundaryFaceHandler, mpit);
    }

    template<typename T, typename S>
    void iterate(const std::function<T(int,int)>& externalCellHandler,
                const std::function<S(int,int)>& internalCellHandler,
                        const std::function<void(int,int,const T&,const S&)>& faceHandler,
                        const std::function<void(int,int)>& boundaryFaceHandler = [](int a, int b){},
                        MPI_Datatype mpit = MPITypeInfer<T>::type()) {
        const auto& realFaceHandler = [&faceHandler, &internalCellHandler](int fid, int cid, const T& tv) {
            faceHandler(fid, cid, tv, internalCellHandler(fid, cid));
        };
        iterate(externalCellHandler, faceHandler, boundaryFaceHandler, mpit);
    }

    template<typename T>
    void iterate(const std::function<T(int,int)>& externalCellHandler,
                        const std::function<void(int,int,const T&)>& faceHandler,
                        const std::function<void(int,int)>& boundaryFaceHandler = [](int a, int b){},
                        MPI_Datatype mpit = MPITypeInfer<T>::type()) {
        iterate_internal(externalCellHandler, faceHandler, boundaryFaceHandler, mpit);
    }

    const PUML<Topo>& puml() const {
        return m_puml;
    }

private:
    template<typename T>
    void iterate_internal(const std::function<T(int,int)>& externalCellHandler,
                        const std::function<void(int,int,const T&)>& faceHandler,
                        const std::function<void(int,int)>& boundaryFaceHandler,
                        MPI_Datatype mpit) {
        int rank = 0, commSize = 1;

        for (int i = 0; i < m_puml.faces().size(); ++i) {
            const auto& face = m_puml.faces()[i];
            int lid[2];
            Upward::cells(m_puml, face, lid);
            assert(!(lid[0] == -1 && lid[1] != -1));

            if (lid[1] != -1) {
                const auto gd1 = externalCellHandler(i, lid[1]);
                faceHandler(i, lid[0], gd1);

                const auto gd0 = externalCellHandler(i, lid[0]);
                faceHandler(i, lid[1], gd0);
            }
            else if (lid[1] == -1 && !face.isShared()) {
                boundaryFaceHandler(i, lid[0]);
            }
        }

#ifdef USE_MPI
        MPI_Comm_rank(m_puml.comm(), &rank);
        MPI_Comm_size(m_puml.comm(), &commSize);

        std::vector<T> transferSend(m_transferDisp[commSize]);
        std::vector<T> transferReceive(m_transferDisp[commSize]);
        for (int i = 0; i < transferSend.size(); ++i) {
            transferSend[i] = externalCellHandler(m_transferFace[i], m_transferCell[i]);
        }

        if (m_sparseComm) {
            int transfer_ranks = 0;
            for (int i = 0, j = 0; i < commSize; ++i) {
                if (m_transferDisp[i+1] > m_transferDisp[i]) {
                    ++transfer_ranks;
                }
            }
            std::vector<MPI_Request> requests(transfer_ranks*2);
            for (int i = 0, j = 0; i < commSize; ++i) {
                if (m_transferDisp[i+1] > m_transferDisp[i]) {
                    MPI_Isend(transferSend.data() + m_transferDisp[i], m_transferSize[i], mpit, i, 0, m_puml.comm(), &requests[2*j]);
                    MPI_Irecv(transferReceive.data() + m_transferDisp[i], m_transferSize[i], mpit, i, 0, m_puml.comm(), &requests[2*j+1]);
                    ++j;
                }
            }
            MPI_Waitall(requests.size(), requests.data(), MPI_STATUS_IGNORE);
        }
        else {
            MPI_Alltoallv(transferSend.data(), m_transferSize.data(), m_transferDisp.data(), mpit, transferReceive.data(), m_transferSize.data(), m_transferDisp.data(), mpit, m_puml.comm());
        }

        for (int i = 0; i < m_transferFace.size(); ++i) {
            const auto& gd1 = transferReceive[i];
            faceHandler(m_transferFace[i], m_transferCell[i], gd1);
        }
#endif
    }

    const PUML<Topo>& m_puml;
#ifdef USE_MPI
    std::vector<int> m_transferSize;
    std::vector<int> m_transferDisp;
    std::vector<int> m_transferCell;
    std::vector<int> m_transferFace;
    bool m_sparseComm;
#endif
};

}

#endif
