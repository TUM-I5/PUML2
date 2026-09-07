// SPDX-FileCopyrightText: 2026 Technical University of Munich
//
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file
 *  This file is part of PUML
 *
 *  For conditions of distribution and use, please see the copyright
 *  notice in the file 'COPYING' at the root directory of this package
 *  and the copyright notice at https://github.com/TUM-I5/PUML
 */

#ifndef PUML_TYPES_H
#define PUML_TYPES_H

#include <cstddef>
#include <cstdint>

namespace PUML {

/**
 * Identifies an entity in the input mesh and across ranks. The width is fixed
 * because it is the width the mesh files use.
 */
using GlobalId = std::uint64_t;

/**
 * Indexes an entity on the rank that holds it.
 */
using LocalId = std::uint32_t;

/**
 * Counts entities, values or bytes.
 */
using Size = std::size_t;

/**
 * Stands for "no such entity" wherever a local id is optional.
 */
constexpr LocalId InvalidLocalId = ~LocalId{0};

// Ranks stay int throughout, because that is what MPI uses for them.

} // namespace PUML

#endif // PUML_TYPES_H
