/*
 * advectiondiffusionproblem_base.hpp
 *
 *  Created on: Jan 24, 2022
 *      Author: heena
 */

#ifndef PROJECT_INCLUDE_ADVECTIONDIFFUSIONPROBLEM_BASE_HPP_
#define PROJECT_INCLUDE_ADVECTIONDIFFUSIONPROBLEM_BASE_HPP_

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector.h>

// Distributed triangulation
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include "advection_field.hpp"
#include "matrix_coeff.hpp"
#include "initial_value.hpp"
#include "dirichlet_bc.hpp"
#include "neumann_bc.hpp"
#include "right_hand_side.hpp"

// STL
#include <cmath>
#include <fstream>
#include <iostream>

// My Headers
#include "config.h"
#include "basis_interface.hpp"

namespace Timedependent_AdvectionDiffusionProblem
{
  template <int dim>
  class BasisInterface;

template <int dim>
class AdvectionDiffusionBase
{
public:
  AdvectionDiffusionBase() = default;

  virtual ~AdvectionDiffusionBase(){};

  virtual const DoFHandler<dim> &
  get_dof_handler() const = 0;

  virtual BasisInterface<dim> *
  get_basis_from_cell_id(CellId cell_id) = 0;
};

}


#endif /* PROJECT_INCLUDE_ADVECTIONDIFFUSIONPROBLEM_BASE_HPP_ */
