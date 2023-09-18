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
class VertexSet
{
public:

	unsigned int capacity;
	unsigned long *vertex_table;
	omp_lock_t *lock;

	VertexSet(unsigned int c)
	{ 
		capacity = c;
		vertex_table = new unsigned long[capacity];
		for(int i = 0; i < capacity; i++)
			vertex_table[i] = -1;
		lock = new omp_lock_t[(capacity / LOCK_SIZE) + 1];
	}

	bool add(unsigned long id)
	{
		unsigned int h = id % capacity;
		while(true)
		{
			omp_set_lock(&lock[h / LOCK_SIZE]);

			if(vertex_table[h] == -1 || vertex_table[h] == id)
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
		bool x = false;
		if(vertex_table[h] == -1)
		{
			vertex_table[h] = id;
			x = true;
		}

		omp_unset_lock(&lock[h / LOCK_SIZE]);

		return x;
	}

	void clear()
	{
		for(unsigned int i = 0; i < (capacity / LOCK_SIZE) + 1; i++)
            omp_destroy_lock(&lock[i]);
        delete [] lock;
		delete [] vertex_table;
	}
};

}

}