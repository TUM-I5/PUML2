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
class FaceMap
{
public:
	/**
	 * Description of a element by the local vertex ids
	 */
	struct Face
	{
		/** The vertices that define this element */
		unsigned int vertices[3];

		Face()
		{ }

		Face(const unsigned int vertices[3])
		{
			memcpy(this->vertices, vertices, 3*sizeof(unsigned int));
			std::sort(this->vertices, this->vertices+3);
		}

		bool operator==(const Face &other) const
		{
			return memcmp(vertices, other.vertices, 3*sizeof(unsigned int)) == 0;
		}
	};

	struct HashItem
	{
		Face face;
		unsigned int id;
		unsigned int cells[2];
	};

	unsigned int capacity;
	HashItem *face_table;
	omp_lock_t *lock;

	FaceMap(unsigned int c)
	{ 
		capacity = c;
		face_table = new HashItem[capacity];
		lock = new omp_lock_t[(capacity / LOCK_SIZE) + 1];
	}

	void add(const unsigned int vertices[3], unsigned int cell)
	{
		const Face f(vertices);
		unsigned int h = hash(f) % capacity;
		int x = 0;
		while(true)
		{
			omp_set_lock(&lock[h / LOCK_SIZE]);
			if(face_table[h].cells[0] == -1 || face_table[h].face == f)
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
		if(face_table[h].cells[0] == -1)
		{
			face_table[h].face = f;
			face_table[h].cells[0] = cell;
			omp_unset_lock(&lock[h / LOCK_SIZE]);
		}
		else
		{
			omp_unset_lock(&lock[h / LOCK_SIZE]);
			if(face_table[h].cells[0] > cell)
			{
				face_table[h].cells[1] = face_table[h].cells[0];
				face_table[h].cells[0] = cell;
			}
			else
				face_table[h].cells[1] = cell;
		}
	}

	unsigned int find(unsigned int vertices[3]) const
	{
		const Face f(vertices);
		unsigned int h = hash(f) % capacity;

		while(face_table[h].cells[0] != -1)
		{
			if(face_table[h].face == f)
				return face_table[h].id;
			h++;
			h %= capacity;
		}
		return -1;
	}

	void clear()
	{
        for(unsigned int i = 0; i < capacity / LOCK_SIZE; i++)
            omp_destroy_lock(&lock[i]);
        delete [] lock;
		delete [] face_table;
	}

private:
	/**
	 * Taken from: https://stackoverflow.com/questions/2590677/how-do-i-combine-hash-values-in-c0x
	 */
	template<typename T>
	static void hash_combine(std::size_t& seed, const T& v)
	{
		std::hash<T> hasher;
		seed ^= hasher(v) + 0x9e3779b9 + (seed<<7) + (seed>>3) + (seed << 8) + (seed >>11);
	}

	std::size_t hash(const Face &face) const
	{
		std::size_t h = std::hash<unsigned int>{}(face.vertices[0]);
		for (unsigned int i = 1; i < 3; i++)
			hash_combine(h, face.vertices[i]);
			return h;
	}
};

}

}