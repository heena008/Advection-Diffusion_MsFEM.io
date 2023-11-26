/*
 * semilagrangian.hpp
 *
 *  Created on: Jun 10, 2020
 *      Author: heena
 */

#ifndef INCLUDE_SEMILAGRANGIAN_HPP_
#define INCLUDE_SEMILAGRANGIAN_HPP_

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/quadrature_point_data.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
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
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <boost/geometry/index/rtree.hpp>
#include <boost/geometry/strategies/strategies.hpp>
#include <deal.II/base/bounding_box.h>
#include <deal.II/base/config.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/boost_adaptors/bounding_box.h>
#include <deal.II/boost_adaptors/point.h>
#include <deal.II/boost_adaptors/segment.h>
#include <memory>

// STL
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>

// My Headers
#include "advection_field.hpp"
#include "advectiondiffusion_basis.hpp"
#include "config.h"
#include "dirichlet_bc.hpp"
#include "initial_value.hpp"
#include "matrix_coeff.hpp"
#include "neumann_bc.hpp"
#include "right_hand_side.hpp"

namespace Coefficients
{
using namespace dealii;

template <int dim>
   class SemiLagrangian
   {
   public:
     SemiLagrangian() = delete;

     /*!
      * Constructor. Does not initialize everything.
      */
     SemiLagrangian(Coefficients::AdvectionField<dim> &_advection_field);

     /*!
      * Copy constructor is deleted.
      */
     SemiLagrangian(const SemiLagrangian &other) = delete;

     /*!
      * Copy constructor is deleted.
      */
     SemiLagrangian<dim> &
     operator=(const SemiLagrangian &other) = delete;

     /*!
      * Destructor.
      */
     ~SemiLagrangian();

     void
           initialize(const double                       _time_step,
                      const unsigned int                 _n_steps_local,
                      Coefficients::AdvectionField<dim> &_advection_field);


     void
     set_time(double _current_time);

     unsigned int
           get_n_trace_backs();

     Point<dim>
     operator()(const Point<dim> &in) const;

     void
         increase_trace_back_counter();

   private:
     /*!
          * Shared pointer to time-dependent vector coefficient (velocity).
          */
         Coefficients::AdvectionField<dim> *advection_field;

         /*!
          * Global time step size.
          */
         double time_step;

         /*!
          * Number of local time steps.
          */
         unsigned int n_steps_local;

         /*!
          * Current time.
          */
         double current_time;

         /*!
          * Guard against mesh traceback from wrong point in time. The starting
          * point MUST be set intentionally.
          */
         bool is_set_current_time;

         /*!
          * Number to indicate how often this local mesh was traced back in time.
          */
         unsigned int n_trace_backs;

         /*!
          * Guard against using uninitialized object.
          */
         bool is_initialized;


   }; // class SemiLagrangian



template <int dim>
SemiLagrangian<dim>::SemiLagrangian(
  Coefficients::AdvectionField<dim> &_advection_field)
  : advection_field(&_advection_field)
    , time_step(0.0)
    , n_steps_local(0)
    , current_time(0)
    , is_set_current_time(false)
    , n_trace_backs(0)
    , is_initialized(false)
{}


template <int dim>
SemiLagrangian<dim>::~SemiLagrangian()
{}

template <int dim>
void
SemiLagrangian<dim>::initialize(
  const double                       _time_step,
  const unsigned int                 _n_steps_local,
  Coefficients::AdvectionField<dim> &_advection_field)
{
  time_step           = _time_step;
  n_steps_local       = _n_steps_local;
  current_time        = 0;
  is_set_current_time = false;
  n_trace_backs       = 0;

  advection_field = &_advection_field;

  is_initialized = true;
}

template <int dim>
Point<dim>
SemiLagrangian<dim>::
operator()(const Point<dim> &in) const
{

	Assert(is_initialized, ExcNotInitialized());
	  Assert(is_set_current_time, ExcNotInitialized());

	  Point<dim> out(in);

	  for (unsigned int j = 0; j < n_steps_local; ++j)
	    {
	      advection_field->set_time(current_time - j * time_step / n_steps_local);
	      out = out - (time_step / n_steps_local) * advection_field->value(out);
	    }

	  return out;
}

template <int dim>
void
SemiLagrangian<
  dim>::increase_trace_back_counter()
{
  is_set_current_time = false;
  n_trace_backs += 1;
}

template <int dim>
void
SemiLagrangian<dim>::set_time(
  double _current_time)
{
  current_time        = _current_time;
  is_set_current_time = true;
}


template <int dim>
unsigned int
SemiLagrangian<dim>::get_n_trace_backs()
{
  return n_trace_backs;
}


} // namespace Timedependent_AdvectionDiffusionProblem

#endif /* INCLUDE_SEMILAGRANGIAN_HPP_ */
