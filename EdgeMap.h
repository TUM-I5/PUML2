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

#include <omp.h>

namespace PUML
{

namespace internal
{

/**
 * Mapps from a list of local vertex ids to the local id of the element
 */
class EdgeMap
{
public:
	/**
	 * Description of a element by the local vertex ids
	 */
	struct Edge
	{
		/** The vertices that define this element */
		unsigned int vertices[2];

		Edge()
		{ }

		Edge(const unsigned int vertices[2])
		{
			memcpy(this->vertices, vertices, 2*sizeof(unsigned int));
			std::sort(this->vertices, this->vertices+2);
		}

		bool operator==(const Edge &other) const
		{
			return memcmp(vertices, other.vertices, 2*sizeof(unsigned int)) == 0;
		}
	};

	struct HashItem
	{
		Edge edge;
		unsigned int id;
		unsigned int faces[CONTAINER_SIZE];
		int size;
		std::set<unsigned int> additionalFaces;
	};

	unsigned int capacity;
	HashItem *edge_table;
	omp_lock_t *lock;

	EdgeMap(unsigned int c)
	{ 
		capacity = c;
		edge_table = new HashItem[capacity];
		lock = new omp_lock_t[(capacity / LOCK_SIZE) + 1];
	}

	void add(const unsigned int vertices[2], unsigned int face1, unsigned int face2)
	{
		const Edge e(vertices);
		unsigned int h = hash(e) % capacity;

		while(true)
		{
			omp_set_lock(&lock[h / LOCK_SIZE]);

			if(edge_table[h].id == -1 || edge_table[h].edge == e)
			{
				break;
			}
			else
			{
				omp_unset_lock(&lock[h / LOCK_SIZE]);
				h++;
				h %= capacity;
			}
		}
		if(edge_table[h].id == -1)
		{
			edge_table[h].edge = e;
			edge_table[h].id = -2;
		}
		bool insert1 = true;
		bool insert2 = true;
		for(unsigned int i = 0; i < edge_table[h].size && i < CONTAINER_SIZE; i++)
		{
			if(edge_table[h].faces[i] == face1)
				insert1 = false;
			if(edge_table[h].faces[i] == face2)
				insert2 = false;
		}
		if(edge_table[h].size < CONTAINER_SIZE && insert1)
		{
			edge_table[h].faces[edge_table[h].size] = face1;
			insert1 = false;
			edge_table[h].size++;
		}

		if(edge_table[h].size < CONTAINER_SIZE && insert2)
		{
			edge_table[h].faces[edge_table[h].size] = face2;
			insert2 = false;
			edge_table[h].size++;
		}

		if(edge_table[h].size >= CONTAINER_SIZE && insert1)
		{
			if(edge_table[h].additionalFaces.insert(face1).second)
				edge_table[h].size++;
		}

		if(edge_table[h].size >= CONTAINER_SIZE && insert2)
		{
			if(edge_table[h].additionalFaces.insert(face2).second)
				edge_table[h].size++;
		}
		
		omp_unset_lock(&lock[h / LOCK_SIZE]);
	}

	unsigned int find(unsigned int vertices[2]) const
	{
		const Edge e(vertices);
		unsigned int h = hash(e) % capacity;

		while(edge_table[h].id != -1)
		{
			if(edge_table[h].edge == e)
				return edge_table[h].id;
			h++;
			h %= capacity;
		}
		return -1;
	}

	void clear()
	{
		delete [] edge_table;
	}

private:
	/**
	 * Taken from: https://stackoverflow.com/questions/2590677/how-do-i-combine-hash-values-in-c0x
	 */
	template<typename T>
	static void hash_combine(std::size_t& seed, const T& v)
	{
		std::hash<T> hasher;
		seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
	}

	std::size_t hash(const Edge &face) const
	{
		std::size_t h = std::hash<unsigned int>{}(face.vertices[0]);
		for (unsigned int i = 1; i < 2; i++)
			hash_combine(h, face.vertices[i]);
			return h;
	}
};

}

}