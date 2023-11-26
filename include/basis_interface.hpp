/*
 * basis_interface.hpp
 *
 *  Created on: Jan 4, 2022
 *      Author: heena
 */

#ifndef INCLUDE_BASIS_INTERFACE_HPP_
#define INCLUDE_BASIS_INTERFACE_HPP_

// Deal.ii
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/numerics/fe_field_function.h>

#include "dirichlet_bc.hpp"
#include "initial_value.hpp"
#include "matrix_coeff.hpp"
#include "neumann_bc.hpp"
#include "right_hand_side.hpp"
#include "advection_field.hpp"
#include "q1basis.hpp"
#include <config.h>
#include "advectiondiffusion_basis.hpp"

namespace Timedependent_AdvectionDiffusionProblem {
using namespace dealii;

template <int dim>
class AdvectionDiffusionBase;

/*!
 * @class BasisInterface
 *
 * Interface class to multiscale basis functions.
 * The idea here is to connect and transfer basis information from fine to coarse.
 */

template <int dim>
class BasisInterface
{
public:
	BasisInterface() = delete;

	BasisInterface(typename Triangulation<dim>::active_cell_iterator &global_cell,
		  	  	  	  	  bool is_first_cell, unsigned int local_subdomain,
						   MPI_Comm mpi_communicator, AdvectionDiffusionBase<dim>& global_problem);

	/*!
	     * Copy constructor is necessary since cell_id-basis pairs will
	     * be copied into a basis std::map. Copying is only possible as
	     * long as large objects are not initialized.
	     */

	BasisInterface(const BasisInterface<dim> &X);
	  /*!
	     * Destructor must be virtual.
	     */

	virtual ~BasisInterface()=0;
	  /*!
	     * Initialization function of the object. Must be called before first time
	     * step update.
	     */

	virtual void initialize() = 0;
	   /*!
	     * Make a global time step.
	     */

	virtual void make_time_step()= 0;
	   /*!
	     * Write out global solution in this cell as vtu.
	     */

	virtual void output_global_solution_in_cell() const = 0;

    /*!
     * Return the multiscale element matrix produced
     * from local basis functions.
     */

	virtual const FullMatrix<double> &
	  get_global_element_matrix(bool current_time_flag) const = 0;
	  /*!
	     * Get the right hand-side that was locally assembled
	     * to speed up the global assembly.
	     */

	virtual const Vector<double> &get_global_element_rhs(bool current_time_flag) const = 0;
	   /*!
	     * Return filename for local pvtu record.
	     */
	virtual const std::string &get_filename_global() const final;

    /*!
     * Get reference to global cell as pointer to DoFCellAcessor.
     */

	 virtual typename DoFHandler<dim>::active_cell_iterator
	    get_global_cell_dof_accessor() final;

	    /*!
	     * Get reference to global cell as pointer to CellAcessor.
	     */

	    virtual typename Triangulation<dim>::active_cell_iterator
	    get_global_cell() final;
	    /*!
	      * Get global cell id.
	      */

	    virtual CellId
	    get_global_cell_id() const final;
	    /*!
	       * For some basis objects an initial reconstruction must be done. The
	       * default implementation in the base class does nothing bit it could be
	       * reimplemented in derived classes.
	       */

	    virtual void
	    initial_reconstruction();
	    /*!
	     * @brief Set global weights.
	     *
	     * The coarse weights of the global solution determine
	     * the local multiscale solution. They must be computed
	     * and then set locally to write an output.
	     */

	virtual void set_global_weights(const std::vector<double> &global_weights)= 0;
	/*!
	     * Get an info string to append to filenames.
	     */

	 virtual const std::string
	    get_basis_info_string() = 0;
	 /*!
	     * Read-only access to global distributed DoFHandler.
	     */

	    virtual const DoFHandler<dim> &
	    get_global_dof_handler() const final;
	    /*!
	     * Get a const reference to locally owned (classic) field function object.
	     */

	      virtual
	        std::shared_ptr<Functions::FEFieldFunction<dim>>
	        get_local_field_function();
	      /*!
	         * Get const reference to other local cell basis from locally owned cell_id
	         */

	      virtual BasisInterface<dim> *
	   	       get_other_local_basis(CellId other_local_cell_id) final;

protected:
    /*!
     * Current MPI communicator.
     */
    MPI_Comm mpi_communicator;

    std::string filename_global;

    /*!
     * Guard for global filename passing without proper value for pvtu-file in
     * global output.
     */
    bool is_set_filename_global;

    /*!
     * Reference to global cell.
     */
    typename Triangulation<dim>::active_cell_iterator global_cell;

    /*!
     * Global cell id.
     */
    const CellId global_cell_id;

    /*!
     * Bool indicating if global_cell is the first in global mesh. Relevant to
     * output.
     */
    const bool is_first_cell;

    const unsigned int local_subdomain;

private:

    /*!
       * Pointer to global problem. This should not be accessible in derived
       * classes.
       */

  AdvectionDiffusionBase<dim>  *global_problem_ptr;

};

template <int dim>
BasisInterface<dim>::BasisInterface(
typename Triangulation<dim>::active_cell_iterator &_global_cell,
  	  bool is_first_cell, unsigned int local_subdomain,
 MPI_Comm mpi_communicator, AdvectionDiffusionBase<dim>& global_problem)
:mpi_communicator(mpi_communicator),
filename_global(""),
is_set_filename_global(false),
global_cell(_global_cell),
global_cell_id(global_cell->id()),
is_first_cell(is_first_cell),
local_subdomain(local_subdomain),
global_problem_ptr(&global_problem)
{}

template <int dim>
BasisInterface<dim>::BasisInterface(const BasisInterface<dim> &X)
:mpi_communicator(X.mpi_communicator),
filename_global(X.filename_global),
is_set_filename_global(X.is_set_filename_global),
global_cell(X.global_cell),
global_cell_id(X.global_cell_id),
is_first_cell(X.is_first_cell),
local_subdomain(X.local_subdomain),
global_problem_ptr(X.global_problem_ptr)
{}

template <int dim>
BasisInterface<dim>::~BasisInterface()
{}

template <int dim>
const std::string &BasisInterface<dim>::get_filename_global() const
{
  Assert(is_set_filename_global,
         ExcMessage("Global filename must be set in each time step."));

  return filename_global;
}

template <int dim>
const DoFHandler<dim> &
BasisInterface<dim>::get_global_dof_handler() const
{

  return global_problem_ptr->get_dof_handler();
}

template <int dim>
  typename Triangulation<dim>::active_cell_iterator
  BasisInterface<dim>::get_global_cell()
  {
    return global_cell;
  }


  template <int dim>
  CellId
  BasisInterface<dim>::get_global_cell_id() const
  {
    return global_cell->id();
  }


  template <int dim>
  typename DoFHandler<dim>::active_cell_iterator
  BasisInterface<dim>::get_global_cell_dof_accessor()
  {
    typename DoFHandler<dim>::active_cell_iterator this_global_cell(
      &get_global_dof_handler().get_triangulation(),
      global_cell->level(),
      global_cell->index(),
      &get_global_dof_handler());

    return this_global_cell;
  }

  template <int dim>
    std::shared_ptr<Functions::FEFieldFunction<dim>>
    BasisInterface<dim>::get_local_field_function()
    {
      AssertThrow(
        false,
        ExcMessage(
          "This function must be called in a derived class that actually possesses a local field function."));

    }



  template <int dim>
  void
  BasisInterface<dim>::initial_reconstruction()
  {}


  template <int dim>
  BasisInterface<dim> *
  BasisInterface<dim>::get_other_local_basis(CellId other_local_cell_id)
  {
    return global_problem_ptr->get_basis_from_cell_id(other_local_cell_id);
  }

} // namespace BasisInterface



#endif /* INCLUDE_BASIS_INTERFACE_HPP_ */
