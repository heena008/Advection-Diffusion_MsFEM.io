/*
 * mpi.cc
 *
 *  Created on: Sep 17, 2020
 *      Author: heena
 */

#include <deal.II/base/exceptions.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi.templates.h>
#include <deal.II/base/mpi_compute_index_owner_internal.h>
#include <deal.II/base/mpi_tags.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector_memory.h>
#include <iostream>
#include <numeric>
#include <set>
#include <vector>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_vector.h>
#include <Epetra_MpiComm.h>
#include <deal.II/lac/petsc_block_vector.h>
#include <deal.II/lac/petsc_vector.h>
#include <petscsys.h>
#include <deal.II/lac/slepc_solver.h>
#include <slepcsys.h>
#include <p4est_bits.h>
#include <zoltan_cpp.h>

#include <algorithm>
#include <bitset>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace dealii;




