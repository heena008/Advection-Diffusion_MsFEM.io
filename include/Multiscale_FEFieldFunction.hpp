/*
 * Multiscale_FEFieldFunction.hpp
 *
 *  Created on: Jan 18, 2022
 *      Author: heena
 */

#ifndef PROJECT_INCLUDE_MULTISCALE_FEFIELDFUNCTION_HPP_
#define PROJECT_INCLUDE_MULTISCALE_FEFIELDFUNCTION_HPP_

// Deal.ii
#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/tria.h>
//#include <deal.II/base/std_cxx17/optional.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/fe_field_function.h>
#include <get_domain_points.hpp>
#include "config.h"
#include "basis_interface.hpp"

using namespace dealii;

template <int dim,
typename DoFHandlerType = DoFHandler<dim>,
bool is_periodic        = true>
class MsFEFieldFunctionMPI : public Function<dim>
{
public:
  /*!
   * Constructor.
   */
	 MsFEFieldFunctionMPI(
	    MPI_Comm                                    mpi_communicator,
		Timedependent_AdvectionDiffusionProblem::BasisInterface<dim> &_local_basis_ptr,
	    const Mapping<dim> &mapping = StaticMappingQ1<dim>::mapping);

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;


private:

  MPI_Comm mpi_communicator;

  SmartPointer<const DoFHandler<dim>, MsFEFieldFunctionMPI<dim>>
    global_dh;

  const Mapping<dim> &mapping;
  typename DoFHandler<dim>::active_cell_iterator this_global_cell;

 Timedependent_AdvectionDiffusionProblem::BasisInterface<dim> *local_basis_ptr;

 Coefficients::UnitCellPointFinder<dim> point_finder;

 /*!
   * Guard against using an uninitialized reconstructor.
   */
  bool is_initialized;

  /*!
   * Flag to trigger debug printing.
   */
  const bool debug_print = true;

};

template <int dim, typename DoFHandlerType, bool is_periodic>
MsFEFieldFunctionMPI<dim, DoFHandlerType, is_periodic>::MsFEFieldFunctionMPI(
  MPI_Comm                                    mpi_communicator,
  Timedependent_AdvectionDiffusionProblem::BasisInterface<dim> &_local_basis,
  const Mapping<dim> &                        _global_mapping)
  : Function<dim>(1)
  , mpi_communicator(mpi_communicator)
  , global_dh(&_local_basis.get_global_dof_handler(), "MsFEFieldFunctionMPI")
  , mapping(_global_mapping)
  , this_global_cell(_local_basis.get_global_cell_dof_accessor())
  , local_basis_ptr(&_local_basis)
  , point_finder()
  , is_initialized(true)
{}


template <int dim, typename DoFHandlerType, bool is_periodic>
double
MsFEFieldFunctionMPI<dim, DoFHandlerType, is_periodic>::value(
  const Point<dim> & point,
  const unsigned int comp) const
{
	Assert(is_initialized, ExcNotInitialized());

	  const Point<dim> p = point_finder(point);

  // Shared pointer
  std::shared_ptr<Functions::FEFieldFunction<dim>> local_field_function_ptr =
    local_basis_ptr->get_local_field_function();

  double value;

  typename DoFHandler<dim>::active_cell_iterator cell = this_global_cell;
  if (cell == global_dh->end())
    cell = global_dh->begin_active();

  /*
       * find_point_owner_rank :
        * Find the MPI rank of the cells that contain the input points in a
        * distributed mesh. If any point is not owned by any mesh cell its return
        * value will be `numbers::invalid_subdomain_id`.
        *
        * @note The query points do not need to be owned locally or in the ghost layer.
        *
        * @note This function can only be used with p4est v2.2 and higher, flat manifolds
        * and requires the settings flag
        * `Settings::communicate_vertices_to_p4est` to be set.
        *
        * @note The algorithm is free of communication.
        *
        * @param[in] points a list of query points
        * @return list of owner ranks
        */
  return value = local_field_function_ptr->find_point_owner_rank.value(point);

}


#endif /* PROJECT_INCLUDE_MULTISCALE_FEFIELDFUNCTION_HPP_ */
