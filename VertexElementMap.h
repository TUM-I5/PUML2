/**
 * @file
 *  This file is part of PUML
 *
 *  For conditions of distribution and use, please see the copyright
 *  notice in the file 'COPYING' at the root directory of this package
 *  and the copyright notice at https://github.com/TUM-I5/PUMGen
 *
 * @copyright 2017 Technische Universitaet Muenchen
 * @author Sebastian Rettenberger <sebastian.rettenberger@tum.de>
 */

#ifndef PUML_VERTEXELEMENTMAP_H
#define PUML_VERTEXELEMENTMAP_H

#include <algorithm>
#include <cstring>
#include <unordered_map>

namespace PUML {

namespace internal {

/**
 * Mapps from a list of local vertex ids to the local id of the element
 */
template <unsigned int N> class VertexElementMap {
  private:
  /**
   * Description of a element by the local vertex ids
   */
  struct Element {
    /** The vertices that define this element */
    unsigned int vertices[N];

    Element(const unsigned int vertices[N]) {
      memcpy(this->vertices, vertices, N * sizeof(unsigned int));
      std::sort(this->vertices, this->vertices + N);
    }

    bool operator==(const Element& other) const {
      return memcmp(vertices, other.vertices, N * sizeof(unsigned int)) == 0;
    }
  };

  struct ElementHash {
    std::size_t operator()(const Element& element) const {
      std::size_t h = std::hash<unsigned int>{}(element.vertices[0]);
      for (unsigned int i = 1; i < N; i++)
        hash_combine(h, element.vertices[i]);

      return h;
    }
  };

  std::unordered_map<Element, unsigned int, ElementHash> m_elements;

  public:
  VertexElementMap() {}

  unsigned int add(const unsigned int vertices[N]) {
    const Element e(vertices);

    typename std::unordered_map<Element, unsigned int, ElementHash>::const_iterator it = m_elements.find(e);
    if (it == m_elements.end()) {
      unsigned int id = m_elements.size();
      it = m_elements.emplace(e, id).first;
    }

    return it->second;
  }

  size_t size() const { return m_elements.size(); }

  int find(unsigned int vertices[N]) const {
    typename std::unordered_map<Element, unsigned int, ElementHash>::const_iterator it = m_elements.find(vertices);

    if (it == m_elements.end())
      return -1;

    return it->second;
  }

  void clear() { m_elements.clear(); }

  private:
  /**
   * Taken from: https://stackoverflow.com/questions/2590677/how-do-i-combine-hash-values-in-c0x
   */
  template <typename T> static void hash_combine(std::size_t& seed, const T& v) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
};

} // namespace internal

} // namespace PUML

#endif // PUML_VERTEXELEMENTMAP_H
