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
#ifdef USE_OPENMP
#include <omp.h>
#endif

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

// these will cause errors when defined as well, as the above types already include them
/*template<> class MPITypeInfer<int8_t> { public: static MPI_Datatype type() { return MPI_INT8_T; } };
template<> class MPITypeInfer<int16_t> { public: static MPI_Datatype type() { return MPI_INT16_T; } };
template<> class MPITypeInfer<int32_t> { public: static MPI_Datatype type() { return MPI_INT32_T; } };
template<> class MPITypeInfer<int64_t> { public: static MPI_Datatype type() { return MPI_INT64_T; } };
template<> class MPITypeInfer<uint8_t> { public: static MPI_Datatype type() { return MPI_UINT8_T; } };
template<> class MPITypeInfer<uint16_t> { public: static MPI_Datatype type() { return MPI_UINT16_T; } };
template<> class MPITypeInfer<uint32_t> { public: static MPI_Datatype type() { return MPI_UINT32_T; } };
template<> class MPITypeInfer<uint64_t> { public: static MPI_Datatype type() { return MPI_UINT64_T; } };*/

// used references for this: https://stackoverflow.com/questions/33618937/trouble-understanding-mpi-type-create-struct https://stackoverflow.com/questions/4496842/pass-method-with-template-arguments-to-a-macro https://stackoverflow.com/questions/3229883/static-member-initialization-in-a-class-template
template<typename X, typename Y> class MPITypeInfer<std::pair<X, Y>> { public: static MPI_Datatype type(){
    if (cache == MPI_DATATYPE_NULL) {
        using pairtype = std::pair<X, Y>; // needed, because comma causes problems with C macros...
        int count = 2;
        int array_of_block_lengths[] = {1, 1};
        MPI_Aint array_of_displacements[] = {offsetof(pairtype, first), offsetof(pairtype, second)};
        MPI_Datatype array_of_types[] = { MPITypeInfer<X>::type(), MPITypeInfer<Y>::type() };
        MPI_Type_create_struct(count, array_of_block_lengths, array_of_displacements, array_of_types, &cache);
        MPI_Type_commit(&cache);
    }
    return cache;
}
private: static MPI_Datatype cache;
};

template<typename X, typename Y> MPI_Datatype MPITypeInfer<std::pair<X, Y>>::cache = MPI_DATATYPE_NULL;

template<typename... X> class MPITypeInfer<std::tuple<X...>> { public: static MPI_Datatype type(){
    if (cache == MPI_DATATYPE_NULL) {
        constexpr size_t N = sizeof...(X);
        int array_of_block_lengths[N];
        MPI_Aint array_of_displacements[N];
        MPI_Datatype array_of_types[N];
        fill<size_t(0), X...>(0, array_of_block_lengths, array_of_displacements, array_of_types);
        MPI_Type_create_struct(N, array_of_block_lengths, array_of_displacements, array_of_types, &cache);
        MPI_Type_commit(&cache);
    }
    return cache;
}
private:
using Tupletype = std::tuple<X...>; // this typedef is needed, because a comma causes problems with C macros
static MPI_Datatype cache;

template<size_t Idx, typename... Y>
static void fill(size_t offset, int* array_of_block_lengths, MPI_Aint* array_of_displacements, MPI_Datatype* array_of_types);

template<size_t Idx>
static void fill(size_t offset, int* array_of_block_lengths, MPI_Aint* array_of_displacements, MPI_Datatype* array_of_types) {
}

template<size_t Idx, typename Y, typename... Z>
static void fill(size_t offset, int* array_of_block_lengths, MPI_Aint* array_of_displacements, MPI_Datatype* array_of_types) {
    // https://stackoverflow.com/questions/55071632/determine-member-offset-of-struct-or-tuple-in-template
    array_of_block_lengths[Idx] = 1;
    array_of_displacements[Idx] = offset;
    array_of_types[Idx] = MPITypeInfer<Y>::type();
    fill<Idx+1, Z...>(offset + sizeof(Y), array_of_block_lengths, array_of_displacements, array_of_types);
}

};

template<typename... X> MPI_Datatype MPITypeInfer<std::tuple<X...>>::cache = MPI_DATATYPE_NULL;

template<typename T, size_t Size> class MPITypeInfer<T[Size]> { public: static MPI_Datatype type(){
    if (cache == MPI_DATATYPE_NULL) {
        MPI_Type_contiguous(Size, MPITypeInfer<T>::type(), &cache);
        MPI_Type_commit(&cache);
    }
    return cache;
}
private: static MPI_Datatype cache;
};

template<typename T, size_t Size> MPI_Datatype MPITypeInfer<T[Size]>::cache = MPI_DATATYPE_NULL;

template<typename T, size_t Size1, size_t Size2> class MPITypeInfer<T[Size1][Size2]> { public: static MPI_Datatype type(){
    if (cache == MPI_DATATYPE_NULL) {
        MPI_Type_contiguous(Size1 * Size2, MPITypeInfer<T>::type(), &cache);
        MPI_Type_commit(&cache);
    }
    return cache;
}
private: static MPI_Datatype cache;
};

template<typename T, size_t Size1, size_t Size2> MPI_Datatype MPITypeInfer<T[Size1][Size2]>::cache = MPI_DATATYPE_NULL;

template<typename T, size_t Size1, size_t Size2, size_t Size3> class MPITypeInfer<T[Size1][Size2][Size3]> { public: static MPI_Datatype type(){
    if (cache == MPI_DATATYPE_NULL) {
        MPI_Type_contiguous(Size1 * Size2 * Size3, MPITypeInfer<T>::type(), &cache);
        MPI_Type_commit(&cache);
    }
    return cache;
}
private: static MPI_Datatype cache;
};

template<typename T, size_t Size1, size_t Size2, size_t Size3> MPI_Datatype MPITypeInfer<T[Size1][Size2][Size3]>::cache = MPI_DATATYPE_NULL;

// if it is needed: implement four- and higher-dimensional C arrays

template<typename T, size_t Size> class MPITypeInfer<std::array<T, Size>> { public: static MPI_Datatype type(){
    if (cache == MPI_DATATYPE_NULL) {
        MPI_Type_contiguous(Size, MPITypeInfer<T>::type(), &cache);
        MPI_Type_commit(&cache);
    }
    return cache;
}
private: static MPI_Datatype cache;
};

template<typename X, size_t Size> MPI_Datatype MPITypeInfer<std::array<X, Size>>::cache = MPI_DATATYPE_NULL;

// if nothing else works out, then just pack everything into a byte object---with all potentially ensuing consequences. Maybe we should emit a warning at some point.
template<typename T> class MPITypeInfer { public: static MPI_Datatype type(){
    if (cache == MPI_DATATYPE_NULL) {
        MPI_Type_contiguous(sizeof(T), MPI_BYTE, &cache);
        MPI_Type_commit(&cache);
        logWarning() << "The MPI datatype of the type" << typeid(T).name() << "could not be inferred automatically, and will therefore be replaced by an adequately-sized byte array datatype.";
    }
    return cache;
}
private: static MPI_Datatype cache;
};

template<typename T> MPI_Datatype MPITypeInfer<T>::cache = MPI_DATATYPE_NULL;
#endif

template<TopoType Topo>
class FaceIterator
{
private:
    class Face;
public:
    FaceIterator(const PUML<Topo>& puml, bool sparse_comm = true) : m_sparse_comm(sparse_comm), m_puml(puml)
    {
        int rank = 0, comm_size = 1;

#ifdef USE_MPI
        MPI_Comm_rank(m_puml.comm(), &rank);
        MPI_Comm_size(m_puml.comm(), &comm_size);

        const auto& faces = puml.faces();

        std::vector<std::vector<int>> transfer(comm_size);
        for (int i = 0; i < faces.size(); ++i) {
            int lid[2];
            Upward::cells(puml, faces[i], lid);
            assert(!(lid[0] == -1 && lid[1] != -1));
            assert(faces[i].shared().size() <= 1);
            if (lid[0] != -1 && faces[i].isShared()) {
                transfer[faces[i].shared()[0]].push_back(i);
            }
        }

        m_transfer_size = std::vector<int>(comm_size);
        m_transfer_disp = std::vector<int>(comm_size+1);
        m_transfer_disp[0] = 0;
        for (int i = 0; i < comm_size; ++i) {
            m_transfer_size[i] = transfer[i].size();
            m_transfer_disp[i+1] = m_transfer_disp[i]+m_transfer_size[i];
            std::sort(transfer[i].begin(), transfer[i].end(),
                [&faces] (int a, int b) -> bool {
                    return faces[a].gid() < faces[b].gid();
                });
        }

        m_transfer_cell = std::vector<int>(m_transfer_disp[comm_size]);
        m_transfer_face = std::vector<int>(m_transfer_disp[comm_size]);
        for (int i = 0, j = 0; i < comm_size; ++i) {
            for (int k = 0; k < m_transfer_size[i]; ++j, ++k) {
                const auto& face = faces[transfer[i][k]];
                int lid[2];
                Upward::cells(puml, face, lid);
                m_transfer_cell[j] = lid[0];
                m_transfer_face[j] = transfer[i][k];
            }
        }
#endif
    }
    
    template<typename T>
    void iterate(const std::vector<T>& cell_data,
                        const std::function<void(int,int,const T&,const T&)>& face_handler,
                        const std::function<void(int,int)>& boundary_face_handler = [](int a, int b){},
                        MPI_Datatype mpit = MPITypeInfer<T>::type()) {
        const auto& cell_handler = [&cell_data](int fid, int cid){return cell_data[cid];};
        iterate(cell_handler, face_handler, boundary_face_handler, mpit);
    }

    template<typename T, typename S>
    void iterate(const std::vector<T>& external_cell_data,
                const std::vector<S>& internal_cell_data,
                        const std::function<void(int,int,const T&,const S&)>& face_handler,
                        const std::function<void(int,int)>& boundary_face_handler = [](int a, int b){},
                        MPI_Datatype mpit = MPITypeInfer<T>::type()) {
        const auto& external_cell_handler = [&external_cell_data](int fid, int cid){return external_cell_data[cid];};
        const auto& internal_cell_handler = [&internal_cell_data](int fid, int cid){return internal_cell_data[cid];};
        iterate(external_cell_handler, internal_cell_handler, face_handler, boundary_face_handler, mpit);
    }

    template<typename T>
    void iterate(const std::vector<T>& external_cell_data,
                        const std::function<void(int,int,const T&)>& face_handler,
                        const std::function<void(int,int)>& boundary_face_handler = [](int a, int b){},
                        MPI_Datatype mpit = MPITypeInfer<T>::type()) {
        const auto& external_cell_handler = [&external_cell_data](int fid, int cid){return external_cell_data[cid];};
        iterate_internal(external_cell_handler, face_handler, boundary_face_handler, mpit);
    }

    template<typename T>
    void iterate(const T* cell_data,
                        const std::function<void(int,int,const T&,const T&)>& face_handler,
                        const std::function<void(int,int)>& boundary_face_handler = [](int a, int b){},
                        MPI_Datatype mpit = MPITypeInfer<T>::type()) {
        const auto& cell_handler = [&cell_data](int fid, int cid){return cell_data[cid];};
        iterate(cell_handler, face_handler, boundary_face_handler, mpit);
    }

    template<typename T, typename S>
    void iterate(const T* external_cell_data,
                const S* internal_cell_data,
                        const std::function<void(int,int,const T&,const S&)>& face_handler,
                        const std::function<void(int,int)>& boundary_face_handler = [](int a, int b){},
                        MPI_Datatype mpit = MPITypeInfer<T>::type()) {
        const auto& external_cell_handler = [&external_cell_data](int fid, int cid){return external_cell_data[cid];};
        const auto& internal_cell_handler = [&internal_cell_data](int fid, int cid){return internal_cell_data[cid];};
        iterate(external_cell_handler, internal_cell_handler, face_handler, boundary_face_handler, mpit);
    }

    template<typename T>
    void iterate(const T* external_cell_data,
                        const std::function<void(int,int,const T&)>& face_handler,
                        const std::function<void(int,int)>& boundary_face_handler = [](int a, int b){},
                        MPI_Datatype mpit = MPITypeInfer<T>::type()) {
        const auto& external_cell_handler = [&external_cell_data](int fid, int cid){return external_cell_data[cid];};
        iterate_internal(external_cell_handler, face_handler, boundary_face_handler, mpit);
    }

    template<typename T>
    void iterate(const std::function<T(int,int)>& cell_handler,
                        const std::function<void(int,int,const T&,const T&)>& face_handler,
                        const std::function<void(int,int)>& boundary_face_handler = [](int a, int b){},
                        MPI_Datatype mpit = MPITypeInfer<T>::type()) {
        iterate(cell_handler, cell_handler, face_handler, boundary_face_handler, mpit);
    }

    template<typename T, typename S>
    void iterate(const std::function<T(int,int)>& external_cell_handler,
                const std::function<S(int,int)>& internal_cell_handler,
                        const std::function<void(int,int,const T&,const S&)>& face_handler,
                        const std::function<void(int,int)>& boundary_face_handler = [](int a, int b){},
                        MPI_Datatype mpit = MPITypeInfer<T>::type()) {
        const auto& real_face_handler = [&face_handler, &internal_cell_handler](int fid, int cid, const T& tv) {
            face_handler(fid, cid, tv, internal_cell_handler(fid, cid));
        };
        iterate(external_cell_handler, face_handler, boundary_face_handler, mpit);
    }

    template<typename T>
    void iterate(const std::function<T(int,int)>& external_cell_handler,
                        const std::function<void(int,int,const T&)>& face_handler,
                        const std::function<void(int,int)>& boundary_face_handler = [](int a, int b){},
                        MPI_Datatype mpit = MPITypeInfer<T>::type()) {
        iterate_internal(external_cell_handler, face_handler, boundary_face_handler, mpit);
    }

    const PUML<Topo>& puml() const {
        return m_puml;
    }

private:
    template<typename T>
    void iterate_internal(const std::function<T(int,int)>& external_cell_handler,
                        const std::function<void(int,int,const T&)>& face_handler,
                        const std::function<void(int,int)>& boundary_face_handler,
                        MPI_Datatype mpit) {
        int rank = 0, comm_size = 1;

        for (int i = 0; i < m_puml.faces().size(); ++i) {
            const auto& face = m_puml.faces()[i];
            int lid[2];
            Upward::cells(m_puml, face, lid);
            assert(!(lid[0] == -1 && lid[1] != -1));

            if (lid[1] != -1) {
                const auto gd1 = external_cell_handler(i, lid[1]);
                face_handler(i, lid[0], gd1);

                const auto gd0 = external_cell_handler(i, lid[0]);
                face_handler(i, lid[1], gd0);
            }
            else if (lid[1] == -1 && !face.isShared()) {
                boundary_face_handler(i, lid[0]);
            }
        }

#ifdef USE_MPI
        MPI_Comm_rank(m_puml.comm(), &rank);
        MPI_Comm_size(m_puml.comm(), &comm_size);

        std::vector<T> transfer_send(m_transfer_disp[comm_size]);
        std::vector<T> transfer_receive(m_transfer_disp[comm_size]);
        for (int i = 0; i < transfer_send.size(); ++i) {
            transfer_send[i] = external_cell_handler(m_transfer_face[i], m_transfer_cell[i]);
        }

        if (m_sparse_comm) {
            int transfer_ranks = 0;
            for (int i = 0, j = 0; i < comm_size; ++i) {
                if (m_transfer_disp[i+1] > m_transfer_disp[i]) {
                    ++transfer_ranks;
                }
            }
            std::vector<MPI_Request> requests(transfer_ranks*2);
            for (int i = 0, j = 0; i < comm_size; ++i) {
                if (m_transfer_disp[i+1] > m_transfer_disp[i]) {
                    MPI_Isend(transfer_send.data() + m_transfer_disp[i], m_transfer_size[i], mpit, i, 0, m_puml.comm(), &requests[2*j]);
                    MPI_Irecv(transfer_receive.data() + m_transfer_disp[i], m_transfer_size[i], mpit, i, 0, m_puml.comm(), &requests[2*j+1]);
                    ++j;
                }
            }
            MPI_Waitall(requests.size(), requests.data(), MPI_STATUS_IGNORE);
        }
        else {
            MPI_Alltoallv(transfer_send.data(), m_transfer_size.data(), m_transfer_disp.data(), mpit, transfer_receive.data(), m_transfer_size.data(), m_transfer_disp.data(), mpit, m_puml.comm());
        }

        for (int i = 0; i < m_transfer_face.size(); ++i) {
            const auto& gd1 = transfer_receive[i];
            face_handler(m_transfer_face[i], m_transfer_cell[i], gd1);
        }
#endif
    }

    const PUML<Topo>& m_puml;
#ifdef USE_MPI
    std::vector<int> m_transfer_size;
    std::vector<int> m_transfer_disp;
    std::vector<int> m_transfer_cell;
    std::vector<int> m_transfer_face;
    bool m_sparse_comm;
#endif
};

}

#endif
