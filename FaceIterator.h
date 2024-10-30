// SPDX-FileCopyrightText: 2019-2023 Technical University of Munich
//
// SPDX-License-Identifier: BSD-3-Clause
/**
 * @file
 *  This file is part of PUML
 *
 *  For conditions of distribution and use, please see the copyright
 *  notice in the file 'COPYING' at the root directory of this package
 *  and the copyright notice at https://github.com/TUM-I5/PUMGen
 *
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
#include <functional>
#include <type_traits>
#include <utility>
#include "Upward.h"
#include "TypeInference.h"

#include "utils/logger.h"

namespace PUML {

template <TopoType Topo>
class FaceIterator {
  private:
  class Face;

  public:
  FaceIterator(const PUML<Topo>& puml, bool sparseComm = true)
      : m_sparseComm(sparseComm), m_puml(puml) {
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
    m_transferDisp = std::vector<int>(commSize + 1);
    m_transferDisp[0] = 0;
    for (int i = 0; i < commSize; ++i) {
      m_transferSize[i] = transfer[i].size();
      m_transferDisp[i + 1] = m_transferDisp[i] + m_transferSize[i];
      std::sort(transfer[i].begin(), transfer[i].end(), [&faces](int a, int b) -> bool {
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

  // FaceHandlerFunc: void(int,int,const T&,const T&)
  template <typename T,
            typename FaceHandlerFunc,
            std::enable_if_t<std::is_invocable_v<FaceHandlerFunc, int, int, const T&, const T&>,
                             bool> = true>
  void forEach(const std::vector<T>& cellData,
               FaceHandlerFunc&& faceHandler,
               MPI_Datatype mpit = MPITypeInfer<T>::type()) {
    auto cellHandler = [&cellData](int fid, int cid) { return cellData[cid]; };
    forEach<T, T>(std::move(cellHandler),
                  std::forward<FaceHandlerFunc>(faceHandler),
                  std::move([](int a, int b) {}),
                  mpit);
  }

  // FaceHandlerFunc: void(int,int,const T&,const T&)
  // BoundaryFaceHandlerFunc: void(int,int)
  template <typename T,
            typename FaceHandlerFunc,
            typename BoundaryFaceHandlerFunc,
            std::enable_if_t<std::is_invocable_v<FaceHandlerFunc, int, int, const T&, const T&>,
                             bool> = true,
            std::enable_if_t<std::is_invocable_v<BoundaryFaceHandlerFunc, int, int>, bool> = true>
  void forEach(const std::vector<T>& cellData,
               FaceHandlerFunc&& faceHandler,
               BoundaryFaceHandlerFunc&& boundaryFaceHandler,
               MPI_Datatype mpit = MPITypeInfer<T>::type()) {
    auto cellHandler = [&cellData](int fid, int cid) { return cellData[cid]; };
    forEach<T, T>(std::move(cellHandler),
                  std::forward<FaceHandlerFunc>(faceHandler),
                  std::forward<BoundaryFaceHandlerFunc>(boundaryFaceHandler),
                  mpit);
  }

  // FaceHandlerFunc: void(int,int,const T&,const S&)
  template <typename T,
            typename S,
            typename FaceHandlerFunc,
            std::enable_if_t<std::is_invocable_v<FaceHandlerFunc, int, int, const T&, const S&>,
                             bool> = true>
  void forEach(const std::vector<T>& externalCellData,
               const std::vector<S>& internalCellData,
               FaceHandlerFunc&& faceHandler,
               MPI_Datatype mpit = MPITypeInfer<T>::type()) {
    auto externalCellHandler = [&externalCellData](int fid, int cid) {
      return externalCellData[cid];
    };
    auto internalCellHandler = [&internalCellData](int fid, int cid) {
      return internalCellData[cid];
    };
    forEach<T, S>(std::move(externalCellHandler),
                  std::move(internalCellHandler),
                  std::forward<FaceHandlerFunc>(faceHandler),
                  std::move([](int a, int b) {}),
                  mpit);
  }

  // FaceHandlerFunc: void(int,int,const T&,const S&)
  // BoundaryFaceHandlerFunc: void(int,int)
  template <typename T,
            typename S,
            typename FaceHandlerFunc,
            typename BoundaryFaceHandlerFunc,
            std::enable_if_t<std::is_invocable_v<FaceHandlerFunc, int, int, const T&, const S&>,
                             bool> = true,
            std::enable_if_t<std::is_invocable_v<BoundaryFaceHandlerFunc, int, int>, bool> = true>
  void forEach(const std::vector<T>& externalCellData,
               const std::vector<S>& internalCellData,
               FaceHandlerFunc&& faceHandler,
               BoundaryFaceHandlerFunc&& boundaryFaceHandler,
               MPI_Datatype mpit = MPITypeInfer<T>::type()) {
    auto externalCellHandler = [&externalCellData](int fid, int cid) {
      return externalCellData[cid];
    };
    auto internalCellHandler = [&internalCellData](int fid, int cid) {
      return internalCellData[cid];
    };
    forEach<T, S>(std::move(externalCellHandler),
                  std::move(internalCellHandler),
                  std::forward<FaceHandlerFunc>(faceHandler),
                  std::forward<BoundaryFaceHandlerFunc>(boundaryFaceHandler),
                  mpit);
  }

  // FaceHandlerFunc: void(int,int,const T&)
  template <typename T,
            typename FaceHandlerFunc,
            std::enable_if_t<std::is_invocable_v<FaceHandlerFunc, int, int, const T&>, bool> = true>
  void forEach(const std::vector<T>& externalCellData,
               FaceHandlerFunc&& faceHandler,
               MPI_Datatype mpit = MPITypeInfer<T>::type()) {
    auto externalCellHandler = [&externalCellData](int fid, int cid) {
      return externalCellData[cid];
    };
    internalforEach<T>(std::move(externalCellHandler),
                       std::forward<FaceHandlerFunc>(faceHandler),
                       std::move([](int a, int b) {}),
                       mpit);
  }

  // FaceHandlerFunc: void(int,int,const T&)
  // BoundaryFaceHandlerFunc: void(int,int)
  template <typename T,
            typename FaceHandlerFunc,
            typename BoundaryFaceHandlerFunc,
            std::enable_if_t<std::is_invocable_v<FaceHandlerFunc, int, int, const T&>, bool> = true,
            std::enable_if_t<std::is_invocable_v<BoundaryFaceHandlerFunc, int, int>, bool> = true>
  void forEach(const std::vector<T>& externalCellData,
               FaceHandlerFunc&& faceHandler,
               BoundaryFaceHandlerFunc&& boundaryFaceHandler,
               MPI_Datatype mpit = MPITypeInfer<T>::type()) {
    auto externalCellHandler = [&externalCellData](int fid, int cid) {
      return externalCellData[cid];
    };
    internalforEach<T>(std::move(externalCellHandler),
                       std::forward<FaceHandlerFunc>(faceHandler),
                       std::forward<BoundaryFaceHandlerFunc>(boundaryFaceHandler),
                       mpit);
  }

  // FaceHandlerFunc: void(int,int,const T&,const T&)
  template <typename T,
            typename FaceHandlerFunc,
            std::enable_if_t<std::is_invocable_v<FaceHandlerFunc, int, int, const T&, const T&>,
                             bool> = true>
  void forEach(const T* cellData,
               FaceHandlerFunc&& faceHandler,
               MPI_Datatype mpit = MPITypeInfer<T>::type()) {
    auto cellHandler = [cellData](int fid, int cid) { return cellData[cid]; };
    forEach<T, T>(std::move(cellHandler),
                  std::forward<FaceHandlerFunc>(faceHandler),
                  std::move([](int a, int b) {}),
                  mpit);
  }

  // FaceHandlerFunc: void(int,int,const T&,const T&)
  // BoundaryFaceHandlerFunc: void(int,int)
  template <typename T,
            typename FaceHandlerFunc,
            typename BoundaryFaceHandlerFunc,
            std::enable_if_t<std::is_invocable_v<FaceHandlerFunc, int, int, const T&, const T&>,
                             bool> = true,
            std::enable_if_t<std::is_invocable_v<BoundaryFaceHandlerFunc, int, int>, bool> = true>
  void forEach(const T* cellData,
               FaceHandlerFunc&& faceHandler,
               BoundaryFaceHandlerFunc&& boundaryFaceHandler,
               MPI_Datatype mpit = MPITypeInfer<T>::type()) {
    auto cellHandler = [cellData](int fid, int cid) { return cellData[cid]; };
    forEach<T, T>(std::move(cellHandler),
                  std::forward<FaceHandlerFunc>(faceHandler),
                  std::forward<BoundaryFaceHandlerFunc>(boundaryFaceHandler),
                  mpit);
  }

  // FaceHandlerFunc: void(int,int,const T&,const S&)
  template <typename T,
            typename S,
            typename FaceHandlerFunc,
            std::enable_if_t<std::is_invocable_v<FaceHandlerFunc, int, int, const T&, const S&>,
                             bool> = true>
  void forEach(const T* externalCellData,
               const S* internalCellData,
               FaceHandlerFunc&& faceHandler,
               MPI_Datatype mpit = MPITypeInfer<T>::type()) {
    auto externalCellHandler = [externalCellData](int fid, int cid) {
      return externalCellData[cid];
    };
    auto internalCellHandler = [internalCellData](int fid, int cid) {
      return internalCellData[cid];
    };
    forEach<T, S>(std::move(externalCellHandler),
                  std::move(internalCellHandler),
                  std::forward<FaceHandlerFunc>(faceHandler),
                  std::move([](int a, int b) {}),
                  mpit);
  }

  // FaceHandlerFunc: void(int,int,const T&,const S&)
  // BoundaryFaceHandlerFunc: void(int,int)
  template <typename T,
            typename S,
            typename FaceHandlerFunc,
            typename BoundaryFaceHandlerFunc,
            std::enable_if_t<std::is_invocable_v<FaceHandlerFunc, int, int, const T&, const S&>,
                             bool> = true,
            std::enable_if_t<std::is_invocable_v<BoundaryFaceHandlerFunc, int, int>, bool> = true>
  void forEach(const T* externalCellData,
               const S* internalCellData,
               FaceHandlerFunc&& faceHandler,
               BoundaryFaceHandlerFunc&& boundaryFaceHandler,
               MPI_Datatype mpit = MPITypeInfer<T>::type()) {
    auto externalCellHandler = [externalCellData](int fid, int cid) {
      return externalCellData[cid];
    };
    auto internalCellHandler = [internalCellData](int fid, int cid) {
      return internalCellData[cid];
    };
    forEach<T, S>(std::move(externalCellHandler),
                  std::move(internalCellHandler),
                  std::forward<FaceHandlerFunc>(faceHandler),
                  std::forward<BoundaryFaceHandlerFunc>(boundaryFaceHandler),
                  mpit);
  }

  // FaceHandlerFunc: void(int,int,const T&)
  template <typename T,
            typename FaceHandlerFunc,
            std::enable_if_t<std::is_invocable_v<FaceHandlerFunc, int, int, const T&>, bool> = true>
  void forEach(const T* externalCellData,
               FaceHandlerFunc&& faceHandler,
               MPI_Datatype mpit = MPITypeInfer<T>::type()) {
    auto externalCellHandler = [externalCellData](int fid, int cid) {
      return externalCellData[cid];
    };
    internalforEach<T>(std::move(externalCellHandler),
                       std::forward<FaceHandlerFunc>(faceHandler),
                       std::move([](int a, int b) {}),
                       mpit);
  }

  // FaceHandlerFunc: void(int,int,const T&)
  // BoundaryFaceHandlerFunc: void(int,int)
  template <typename T,
            typename FaceHandlerFunc,
            typename BoundaryFaceHandlerFunc,
            std::enable_if_t<std::is_invocable_v<FaceHandlerFunc, int, int, const T&>, bool> = true,
            std::enable_if_t<std::is_invocable_v<BoundaryFaceHandlerFunc, int, int>, bool> = true>
  void forEach(const T* externalCellData,
               FaceHandlerFunc&& faceHandler,
               BoundaryFaceHandlerFunc&& boundaryFaceHandler,
               MPI_Datatype mpit = MPITypeInfer<T>::type()) {
    auto externalCellHandler = [externalCellData](int fid, int cid) {
      return externalCellData[cid];
    };
    internalforEach<T>(std::move(externalCellHandler),
                       std::forward<FaceHandlerFunc>(faceHandler),
                       std::forward<BoundaryFaceHandlerFunc>(boundaryFaceHandler),
                       mpit);
  }

  // CellHandlerFunc: T(int,int)
  // FaceHandlerFunc: void(int,int,const T&,const T&)
  template <typename T,
            typename CellHandlerFunc,
            typename FaceHandlerFunc,
            std::enable_if_t<std::is_invocable_r_v<T, CellHandlerFunc, int, int>, bool> = true,
            std::enable_if_t<std::is_invocable_v<FaceHandlerFunc, int, int, const T&, const T&>,
                             bool> = true>
  void forEach(CellHandlerFunc&& cellHandler,
               FaceHandlerFunc&& faceHandler,
               MPI_Datatype mpit = MPITypeInfer<T>::type()) {
    // no direct move/forward possible here for cellHandler
    auto externalCellHandler = [&cellHandler](int fid, int cid) {
      return std::invoke(cellHandler, fid, cid);
    };
    auto internalCellHandler = [&cellHandler](int fid, int cid) {
      return std::invoke(cellHandler, fid, cid);
    };
    forEach<T, T>(std::move(externalCellHandler),
                  std::move(internalCellHandler),
                  std::forward<FaceHandlerFunc>(faceHandler),
                  std::move([](int a, int b) {}),
                  mpit);
  }

  // CellHandlerFunc: T(int,int)
  // FaceHandlerFunc: void(int,int,const T&,const T&)
  // BoundaryFaceHandlerFunc: void(int,int)
  template <typename T,
            typename CellHandlerFunc,
            typename FaceHandlerFunc,
            typename BoundaryFaceHandlerFunc,
            std::enable_if_t<std::is_invocable_r_v<T, CellHandlerFunc, int, int>, bool> = true,
            std::enable_if_t<std::is_invocable_v<FaceHandlerFunc, int, int, const T&, const T&>,
                             bool> = true,
            std::enable_if_t<std::is_invocable_v<BoundaryFaceHandlerFunc, int, int>, bool> = true>
  void forEach(CellHandlerFunc&& cellHandler,
               FaceHandlerFunc&& faceHandler,
               BoundaryFaceHandlerFunc&& boundaryFaceHandler,
               MPI_Datatype mpit = MPITypeInfer<T>::type()) {
    // no direct move/forward possible here for cellHandler
    auto externalCellHandler = [&cellHandler](int fid, int cid) {
      return std::invoke(cellHandler, fid, cid);
    };
    auto internalCellHandler = [&cellHandler](int fid, int cid) {
      return std::invoke(cellHandler, fid, cid);
    };
    forEach<T, T>(std::move(externalCellHandler),
                  std::move(internalCellHandler),
                  std::forward<FaceHandlerFunc>(faceHandler),
                  std::forward<BoundaryFaceHandlerFunc>(boundaryFaceHandler),
                  mpit);
  }

  // ExternalCellHandlerFunc: T(int,int)
  // InternalCellHandlerFunc: S(int,int)
  // FaceHandlerFunc: void(int,int,const T&,const S&)
  template <
      typename T,
      typename S,
      typename ExternalCellHandlerFunc,
      typename InternalCellHandlerFunc,
      typename FaceHandlerFunc,
      std::enable_if_t<std::is_invocable_r_v<T, ExternalCellHandlerFunc, int, int>, bool> = true,
      std::enable_if_t<std::is_invocable_r_v<S, InternalCellHandlerFunc, int, int>, bool> = true,
      std::enable_if_t<std::is_invocable_v<FaceHandlerFunc, int, int, const T&, const S&>, bool> =
          true>
  void forEach(ExternalCellHandlerFunc&& externalCellHandler,
               InternalCellHandlerFunc&& internalCellHandler,
               FaceHandlerFunc&& faceHandler,
               MPI_Datatype mpit = MPITypeInfer<T>::type()) {
    auto realFaceHandler = [faceHandler = std::forward<FaceHandlerFunc>(faceHandler),
                            internalCellHandler = std::forward<InternalCellHandlerFunc>(
                                internalCellHandler)](int fid, int cid, const T& tv) {
      std::invoke(faceHandler, fid, cid, tv, std::invoke(internalCellHandler, fid, cid));
    };
    forEach<T>(std::forward<ExternalCellHandlerFunc>(externalCellHandler),
               std::move(realFaceHandler),
               std::move([](int a, int b) {}),
               mpit);
  }

  // ExternalCellHandlerFunc: T(int,int)
  // InternalCellHandlerFunc: S(int,int)
  // FaceHandlerFunc: void(int,int,const T&,const S&)
  // BoundaryFaceHandlerFunc: void(int,int)
  template <
      typename T,
      typename S,
      typename ExternalCellHandlerFunc,
      typename InternalCellHandlerFunc,
      typename FaceHandlerFunc,
      typename BoundaryFaceHandlerFunc,
      std::enable_if_t<std::is_invocable_r_v<T, ExternalCellHandlerFunc, int, int>, bool> = true,
      std::enable_if_t<std::is_invocable_r_v<S, InternalCellHandlerFunc, int, int>, bool> = true,
      std::enable_if_t<std::is_invocable_v<FaceHandlerFunc, int, int, const T&, const S&>, bool> =
          true,
      std::enable_if_t<std::is_invocable_v<BoundaryFaceHandlerFunc, int, int>, bool> = true>
  void forEach(ExternalCellHandlerFunc&& externalCellHandler,
               InternalCellHandlerFunc&& internalCellHandler,
               FaceHandlerFunc&& faceHandler,
               BoundaryFaceHandlerFunc&& boundaryFaceHandler,
               MPI_Datatype mpit = MPITypeInfer<T>::type()) {
    auto realFaceHandler = [faceHandler = std::forward<FaceHandlerFunc>(faceHandler),
                            internalCellHandler = std::forward<InternalCellHandlerFunc>(
                                internalCellHandler)](int fid, int cid, const T& tv) {
      std::invoke(faceHandler, fid, cid, tv, std::invoke(internalCellHandler, fid, cid));
    };
    forEach<T>(std::forward<ExternalCellHandlerFunc>(externalCellHandler),
               std::move(realFaceHandler),
               std::forward<BoundaryFaceHandlerFunc>(boundaryFaceHandler),
               mpit);
  }

  // ExternalCellHandlerFunc: T(int,int)
  // FaceHandlerFunc: void(int,int,const T&)
  template <
      typename T,
      typename ExternalCellHandlerFunc,
      typename FaceHandlerFunc,
      std::enable_if_t<std::is_invocable_r_v<T, ExternalCellHandlerFunc, int, int>, bool> = true,
      std::enable_if_t<std::is_invocable_v<FaceHandlerFunc, int, int, const T&>, bool> = true>
  void forEach(ExternalCellHandlerFunc&& externalCellHandler,
               FaceHandlerFunc&& faceHandler,
               MPI_Datatype mpit = MPITypeInfer<T>::type()) {
    internalforEach<T>(std::forward<ExternalCellHandlerFunc>(externalCellHandler),
                       std::forward<FaceHandlerFunc>(faceHandler),
                       std::move([](int a, int b) {}),
                       mpit);
  }

  // ExternalCellHandlerFunc: T(int,int)
  // FaceHandlerFunc: void(int,int,const T&)
  // BoundaryFaceHandlerFunc: void(int,int)
  template <
      typename T,
      typename ExternalCellHandlerFunc,
      typename FaceHandlerFunc,
      typename BoundaryFaceHandlerFunc,
      std::enable_if_t<std::is_invocable_r_v<T, ExternalCellHandlerFunc, int, int>, bool> = true,
      std::enable_if_t<std::is_invocable_v<FaceHandlerFunc, int, int, const T&>, bool> = true,
      std::enable_if_t<std::is_invocable_v<BoundaryFaceHandlerFunc, int, int>, bool> = true>
  void forEach(ExternalCellHandlerFunc&& externalCellHandler,
               FaceHandlerFunc&& faceHandler,
               BoundaryFaceHandlerFunc&& boundaryFaceHandler,
               MPI_Datatype mpit = MPITypeInfer<T>::type()) {
    internalforEach<T>(std::forward<ExternalCellHandlerFunc>(externalCellHandler),
                       std::forward<FaceHandlerFunc>(faceHandler),
                       std::forward<BoundaryFaceHandlerFunc>(boundaryFaceHandler),
                       mpit);
  }

  const PUML<Topo>& puml() const { return m_puml; }

  private:
  // ExternalCellHandlerFunc: T(int,int)
  // FaceHandlerFunc: void(int,int,const T&)
  // BoundaryFaceHandlerFunc: void(int,int)
  template <
      typename T,
      typename ExternalCellHandlerFunc,
      typename FaceHandlerFunc,
      typename BoundaryFaceHandlerFunc,
      std::enable_if_t<std::is_invocable_r_v<T, ExternalCellHandlerFunc, int, int>, bool> = true,
      std::enable_if_t<std::is_invocable_v<FaceHandlerFunc, int, int, const T&>, bool> = true,
      std::enable_if_t<std::is_invocable_v<BoundaryFaceHandlerFunc, int, int>, bool> = true>
  void internalforEach(ExternalCellHandlerFunc&& externalCellHandler,
                       FaceHandlerFunc&& faceHandler,
                       BoundaryFaceHandlerFunc&& boundaryFaceHandler,
                       MPI_Datatype mpit) {
    int rank = 0, commSize = 1;

    for (int i = 0; i < m_puml.faces().size(); ++i) {
      const auto& face = m_puml.faces()[i];
      int lid[2];
      Upward::cells(m_puml, face, lid);
      assert(!(lid[0] == -1 && lid[1] != -1));

      if (lid[1] != -1) {
        const auto gd1 = std::invoke(externalCellHandler, i, lid[1]);
        std::invoke(faceHandler, i, lid[0], gd1);

        const auto gd0 = std::invoke(externalCellHandler, i, lid[0]);
        std::invoke(faceHandler, i, lid[1], gd0);
      } else if (lid[1] == -1 && !face.isShared()) {
        std::invoke(boundaryFaceHandler, i, lid[0]);
      }
    }

#ifdef USE_MPI
    MPI_Comm_rank(m_puml.comm(), &rank);
    MPI_Comm_size(m_puml.comm(), &commSize);

    std::vector<T> transferSend(m_transferDisp[commSize]);
    std::vector<T> transferReceive(m_transferDisp[commSize]);
    for (int i = 0; i < transferSend.size(); ++i) {
      transferSend[i] = std::invoke(externalCellHandler, m_transferFace[i], m_transferCell[i]);
    }

    if (m_sparseComm) {
      int transfer_ranks = 0;
      for (int i = 0, j = 0; i < commSize; ++i) {
        if (m_transferDisp[i + 1] > m_transferDisp[i]) {
          ++transfer_ranks;
        }
      }
      std::vector<MPI_Request> requests(transfer_ranks * 2);
      for (int i = 0, j = 0; i < commSize; ++i) {
        if (m_transferDisp[i + 1] > m_transferDisp[i]) {
          MPI_Isend(static_cast<T*>(transferSend.data()) + m_transferDisp[i],
                    m_transferSize[i],
                    mpit,
                    i,
                    0,
                    m_puml.comm(),
                    &requests[2 * j]);
          MPI_Irecv(static_cast<T*>(transferReceive.data()) + m_transferDisp[i],
                    m_transferSize[i],
                    mpit,
                    i,
                    0,
                    m_puml.comm(),
                    &requests[2 * j + 1]);
          ++j;
        }
      }
      MPI_Waitall(requests.size(), requests.data(), MPI_STATUS_IGNORE);
    } else {
      MPI_Alltoallv(transferSend.data(),
                    m_transferSize.data(),
                    m_transferDisp.data(),
                    mpit,
                    transferReceive.data(),
                    m_transferSize.data(),
                    m_transferDisp.data(),
                    mpit,
                    m_puml.comm());
    }

    for (int i = 0; i < m_transferFace.size(); ++i) {
      const auto& gd1 = transferReceive[i];
      std::invoke(faceHandler, m_transferFace[i], m_transferCell[i], gd1);
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

} // namespace PUML

#endif
