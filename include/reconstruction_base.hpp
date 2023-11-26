/*
 * inverse_matrix.hpp
 *
 *  Created on: Jul 1, 2020
 *      Author: heena
 */

#ifndef INCLUDE_RECONSTRUCTION_BASE_HPP_
#define INCLUDE_RECONSTRUCTION_BASE_HPP_

// Deal.ii
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>


// STL
#include <memory>
#include <stdexcept>
#include <vector>

// My headers
#include "matrix_coeff.hpp"
#include "right_hand_side.hpp"
#include "neumann_bc.hpp"
#include "dirichlet_bc.hpp"
#include "q1basis.hpp"
#include "advectiondiffusion_basis.hpp"
#include "advection_field.hpp"
#include "semilagrangian.hpp"
#include "config.h"
#include "basis_interface.hpp"
#include "reconstruction_assembler.hpp"
//#include "H1_conformal.hpp"


namespace Timedependent_AdvectionDiffusionProblem {
using namespace dealii;


  template <int dim,  class ReconstructionType>
  class SemiLagrangeBasis : public BasisInterface<dim>
  {
  public:
    /*!
     * Delete standard constructor.
     */
    SemiLagrangeBasis() = delete;

    /*!
     * Constructor.
     */
    SemiLagrangeBasis(
      typename Triangulation<dim>::active_cell_iterator &_global_cell,
      bool                                               _is_first_cell,
      unsigned int                                       _local_subdomain,
	  const double                                       _theta,
      MPI_Comm                                           _mpi_communicator,
	  AdvectionDiffusionBase<dim> &                      _global_problem);

    /*!
     * Copy constructor is necessary since cell_id-basis pairs will
     * be copied into a basis std::map. Copying is only possible as
     * long as large objects are not initialized.
     */
    SemiLagrangeBasis(const SemiLagrangeBasis<dim, ReconstructionType> &X);

    /*!
     * Destructor.
     */
    ~SemiLagrangeBasis();

    /*!
     * Initialization function of the object. Must be called before first time
     * step update.
     */
    void
    initialize();

    /*!
     * Initial basis reconstruction.
     */
    virtual void
    initial_reconstruction();

    /*!
     * Make single time step. Note that no reconstruction is done for this
     * class.
     */
    void
    make_time_step();

    /*!
     * Write out global solution in this cell as vtu.
     */
    void
    output_global_solution_in_cell() const override;

    /*!
     * Return the multiscale element matrix produced
     * from local basis functions.
     */
    const FullMatrix<double> &
    get_global_element_matrix(bool current_time_flag) const override;

    /*!
     * Get the right hand-side that was locally assembled
     * to speed up the global assembly.
     */
    const Vector<double> &
    get_global_element_rhs(bool current_time_flag) const override;

    void
    set_global_weights(const std::vector<double> &global_weights) override;

    /*!
     * Get an info string to append to filenames.
     */
    virtual const std::string
    get_basis_info_string() override;

    /*!
     * Get a const reference to locally owned (classic) field function object.
     */
    virtual std::shared_ptr<Functions::FEFieldFunction<dim>>
    get_local_field_function() override;

  private:

    void
    make_grid();

    void
    setup_system();

    void
    output_basis(const std::vector<Vector<double>> &solution_vector) const;

    void
    set_filename_global();

    const unsigned int n_refine_local     = 4;
    const bool         verbose            = false;
    const bool         verbose_all        = false;
    const bool         use_direct_solver  = true;
    const bool         output_first_basis = true;
    const std::string  filename_global_base;

    /*!
     * Local triangulation.
     */
    Triangulation<dim> triangulation;

    FE_Q<dim> fe;

    DoFHandler<dim> dof_handler;

    std::vector<Point<dim>> corner_points;

    /*!
     * Solution vector.
     */
    std::vector<Vector<double>> solution_vector;

    /*!
     * Solution vector at previous time step.
     */
    std::vector<Vector<double>> solution_vector_old;

    /*!
     * Solution vector time derivative.
     */
    std::vector<Vector<double>> solution_vector_time_derivative;

    /*!
     * Flag indicating whether systems for the basis
     * was solved in each time step.
     */
    bool is_solved;

    /*!
     * Contains the right-hand side.
     */
    Vector<double>
      global_rhs; // this is only for the global assembly (speed-up)

    /*!
     * Contains the right-hand side.
     */
    Vector<double>
      global_rhs_old; // this is only for the global assembly (speed-up)

    /*!
     * Holds global multiscale element matrix.
     */
    FullMatrix<double> global_element_matrix;

    /*!
     * Holds global multiscale element matrix at previous time step.
     */
    FullMatrix<double> global_element_matrix_old;

    /*!
     * Holds global multiscale element right-hand side.
     */
    Vector<double> global_element_rhs;

    /*!
     * Holds global multiscale element right-hand side.
     */
    Vector<double> global_element_rhs_old;

    /*!
     * Weights of multiscale basis functions at current time step.
     * The weights are correct if is_set_global_weights=true.
     */
    std::vector<double> global_weights;

    /*!
     * Bool to guard against writing global solution is when global weights are
     * uninitialized.
     */
    bool is_set_global_weights;

    /*!
     * Global solution.
     */
    Vector<double> global_solution;

    /*!
     * Make sure basis object is initialized before time stepping.
     */
    bool is_set_global_solution;

    /*!
     * Make sure basis object is initialized before time stepping.
     */
    bool is_initialized;

    double       time;
    double       time_step;
    unsigned int timestep_number;

    /*!
     * Parameter to determine the "implicitness" of the method.
     * Zero is fully implicit and one is (almost explicit).
     */
    const double theta;

    /*!
     * Time-dependent matrix coefficient (diffusion).
     */
    Coefficients::MatrixCoeff<dim> matrix_coeff;

    /*!
     * Time-dependent vector coefficient (velocity).
     */
    Coefficients::AdvectionField<dim> advection_field;

    /*!
     * Time-dependent scalar coefficient (forcing).
     */
    Coefficients::RightHandSide<dim> right_hand_side;

    /*!
     * Initial conditions
     */
    Coefficients::InitialValue<dim> initial_value;


    ReconstructionType basis_reconstructor;

    std::shared_ptr<Functions::FEFieldFunction<dim>> local_field_function_ptr;

  };


  template <int dim, class ReconstructionType>
SemiLagrangeBasis<dim, ReconstructionType>::SemiLagrangeBasis(
    typename Triangulation<dim>::active_cell_iterator &_global_cell,
    bool                                               _is_first_cell,
    unsigned int                                       _local_subdomain,
	 const double                                       _theta,
    MPI_Comm                                           _mpi_communicator,
    AdvectionDiffusionBase<dim> &                      _global_problem)
    : BasisInterface<dim>(_global_cell,
                                _is_first_cell,
                                _local_subdomain,
                                _mpi_communicator,
                                _global_problem)
    , filename_global_base((dim == 2 ? "solution-ms_2d" : "solution-ms_3d"))
    , fe(1)
    , dof_handler(triangulation)
    , corner_points(GeometryInfo<dim>::vertices_per_cell)
    , solution_vector(GeometryInfo<dim>::vertices_per_cell)
    , solution_vector_old(GeometryInfo<dim>::vertices_per_cell)
    , solution_vector_time_derivative(GeometryInfo<dim>::vertices_per_cell)
    , is_solved(false)
    , global_element_matrix(fe.dofs_per_cell, fe.dofs_per_cell)
    , global_element_matrix_old(fe.dofs_per_cell, fe.dofs_per_cell)
    , global_element_rhs(fe.dofs_per_cell)
    , global_element_rhs_old(fe.dofs_per_cell)
    , global_weights(fe.dofs_per_cell, 0)
    , is_set_global_weights(false)
    , is_set_global_solution(false)
    , is_initialized(false)
    , time(0.0)
    , time_step(1. / 100)
    , timestep_number(0)
    /*
     * theta=1 is implicit Euler,
     * theta=0 is explicit Euler,
     * theta=0.5 is Crank-Nicolson
     */
,theta(_theta)
    , basis_reconstructor(*this,
            _mpi_communicator,
            fe,
            matrix_coeff,
            advection_field,
            right_hand_side,
            solution_vector,
            solution_vector_old,
            solution_vector_time_derivative,
            _is_first_cell)
  {
    // set corner points
    for (unsigned int vertex_n = 0;
         vertex_n < GeometryInfo<dim>::vertices_per_cell;
         ++vertex_n)
      {
        corner_points[vertex_n] = _global_cell->vertex(vertex_n);
      }

    /*
     * Already create a coarse grid. Do not refine here yet!
     */
    GridGenerator::general_cell(triangulation,
                                corner_points,
                                /* colorize faces */ false);

    /*
     * Copy the triangulation into the reconstructor since this guy needs its own
     * copy.
     */
    basis_reconstructor.copy_triangulation(triangulation);
  }




template <int dim, class ReconstructionType>
SemiLagrangeBasis<dim, ReconstructionType>::SemiLagrangeBasis(
  const SemiLagrangeBasis<dim, ReconstructionType> &X)
  : BasisInterface<dim>(X)
  , n_refine_local(X.n_refine_local)
  , verbose(X.verbose)
  , verbose_all(X.verbose_all)
  , use_direct_solver(X.use_direct_solver)
  , output_first_basis(X.output_first_basis)
  , filename_global_base(X.filename_global_base)
  , triangulation()
  , fe(X.fe)
  , dof_handler(triangulation)
  , corner_points(X.corner_points)
  , solution_vector(X.solution_vector)
  , solution_vector_old(X.solution_vector_old)
  , solution_vector_time_derivative(X.solution_vector_time_derivative)
  , is_solved(X.is_solved)
  , global_rhs(X.global_rhs)
  , global_rhs_old(X.global_rhs_old)
  , global_element_matrix(X.global_element_matrix)
  , global_element_matrix_old(X.global_element_matrix_old)
  , global_element_rhs(X.global_element_rhs)
  , global_element_rhs_old(X.global_element_rhs_old)
  , global_weights(X.global_weights)
  , is_set_global_weights(X.is_set_global_weights)
  , global_solution(X.global_solution)
  , is_set_global_solution(X.is_set_global_solution)
  , is_initialized(X.is_initialized)
  , time(X.time)
  , time_step(X.time_step)
  , timestep_number(X.timestep_number)
  /*
   * theta=1 is implicit Euler,
   * theta=0 is explicit Euler,
   * theta=0.5 is Crank-Nicolson
   */
  ,theta(X.theta)
  , matrix_coeff(X.matrix_coeff)
  , advection_field(X.advection_field)
  , right_hand_side(X.right_hand_side)
  , initial_value(X.initial_value)
  , basis_reconstructor(*this,
                        this->mpi_communicator,
                        fe,
                        matrix_coeff,
                        advection_field,
                        right_hand_side,
                        solution_vector,
                        solution_vector_old,
                        solution_vector_time_derivative,
                        this->is_first_cell)
{
  /*
   * Deliberately copy the triangulation. Note that in the used
   * cases the triangulation is not refined and hence small.
   */
  triangulation.copy_triangulation(X.triangulation);

  /*
   * Copy the triangulation into the reconstructor since this guy needs its own
   * copy.
   */
  basis_reconstructor.copy_triangulation(triangulation);
}


template <int dim, class ReconstructionType>
SemiLagrangeBasis<dim, ReconstructionType>::~SemiLagrangeBasis()
{}


template <int dim, class ReconstructionType>
void
SemiLagrangeBasis<dim, ReconstructionType>::make_grid()
{
  triangulation.refine_global(n_refine_local);
}


template <int dim, class ReconstructionType>
void
SemiLagrangeBasis<dim, ReconstructionType>::setup_system()
{
  dof_handler.distribute_dofs(fe);

  for (unsigned int index_basis = 0;
       index_basis < GeometryInfo<dim>::vertices_per_cell;
       ++index_basis)
    {
      solution_vector[index_basis].reinit(dof_handler.n_dofs());
      solution_vector_old[index_basis].reinit(dof_handler.n_dofs());
      solution_vector_time_derivative[index_basis].reinit(dof_handler.n_dofs());
    }

  global_rhs.reinit(dof_handler.n_dofs());
  global_rhs_old.reinit(dof_handler.n_dofs());
  global_solution.reinit(dof_handler.n_dofs());
}


template <int dim, class ReconstructionType>
const FullMatrix<double> &
SemiLagrangeBasis<dim, ReconstructionType>::
  get_global_element_matrix(bool current_time_flag) const
{
  if (current_time_flag)
    {
      return global_element_matrix;
    }
  else
    {
      return global_element_matrix_old;
    }
}


template <int dim, class ReconstructionType>
const Vector<double> &
SemiLagrangeBasis<dim, ReconstructionType>::
  get_global_element_rhs(bool current_time_flag) const
{
  if (current_time_flag)
    {
      return global_element_rhs;
    }
  else
    {
      return global_element_rhs_old;
    }
}


template <int dim, class ReconstructionType>
void
SemiLagrangeBasis<dim, ReconstructionType>::output_basis(
  const std::vector<Vector<double>> &solution_vector) const
{
  Timer timer;
  if (verbose_all)
    {
      std::cout << "		Writing local basis in cell   "
                << this->global_cell_id.to_string() << ".....";

      timer.restart();
    }

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);

  for (unsigned int index_basis = 0;
       index_basis < GeometryInfo<dim>::vertices_per_cell;
       ++index_basis)
    {
      std::vector<std::string> solution_names(
        1, "u_" + Utilities::int_to_string(index_basis, 1));
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        interpretation(1, DataComponentInterpretation::component_is_scalar);

      data_out.add_data_vector(solution_vector[index_basis],
                               solution_names,
                               DataOut<dim>::type_dof_data,
                               interpretation);
    }

  data_out.build_patches();

  // filename
  std::string filename = "basis_semiLagrange";
  filename += "." + Utilities::int_to_string(this->local_subdomain, 5);
  filename += ".cell-" + this->global_cell_id.to_string();
  filename += ".time_step-";
  filename += Utilities::to_string(timestep_number, 4);
  filename += ".vtu";

  std::ofstream output(filename);
  data_out.write_vtu(output);

  if (verbose_all)
    {
      timer.stop();
      std::cout << "done in   " << timer.cpu_time() << "   seconds."
                << std::endl;
    }
}


template <int dim, class ReconstructionType>
const std::string
SemiLagrangeBasis<dim, ReconstructionType>::get_basis_info_string()
{
  return "_semi_lagrange_basis" + basis_reconstructor.get_info_string();
}


template <int dim, class ReconstructionType>
void
SemiLagrangeBasis<dim, ReconstructionType>::set_filename_global()
{
  this->filename_global = filename_global_base + get_basis_info_string() + "." +
                          Utilities::int_to_string(this->local_subdomain, 5) +
                          ".cell-" + this->global_cell_id.to_string() +
                          ".theta-" + Utilities::to_string(theta, 4) +
                          ".time_step-" +
                          Utilities::int_to_string(timestep_number, 4) + ".vtu";

  this->is_set_filename_global = true;
}


template <int dim, class ReconstructionType>
void
SemiLagrangeBasis<dim, ReconstructionType>::
  output_global_solution_in_cell() const
{
  Assert(is_set_global_weights,
         ExcMessage("Global weights must be set in each time step."));

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);

  std::vector<std::string> solution_names(1, "u");

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(
      1, DataComponentInterpretation::component_is_scalar);

  data_out.add_data_vector(global_solution,
                           solution_names,
                           DataOut<dim>::type_dof_data,
                           data_component_interpretation);

  Vector<float> this_subdomain(triangulation.n_active_cells());
  for (unsigned int i = 0; i < this_subdomain.size(); ++i)
    {
      this_subdomain(i) = this->local_subdomain;
    }
  data_out.add_data_vector(this_subdomain, "subdomain");

  // Postprocess
  //      std::unique_ptr<Q_PostProcessor> postprocessor(
  //        new Q_PostProcessor(parameter_filename));
  //      data_out.add_data_vector(global_solution, *postprocessor);

  data_out.build_patches();

  std::ofstream output(this->filename_global);
  data_out.write_vtu(output);
}


template <int dim, class ReconstructionType>
void
SemiLagrangeBasis<dim, ReconstructionType>::set_global_weights(
  const std::vector<double> &weights)
{
  // Copy assignment of global weights
  global_weights = weights;

  // reinitialize the global solution on this cell
  global_solution.reinit(dof_handler.n_dofs());

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  // Set global solution using the weights and the local basis.
  for (unsigned int index_basis = 0; index_basis < dofs_per_cell; ++index_basis)
    {
      // global_solution = 1*global_solution +
      // global_weights[index_basis]*solution_vector[index_basis]
      global_solution.sadd(1,
                           global_weights[index_basis],
                           solution_vector[index_basis]);
    }

//  local_field_function_ptr.reset(
//    new Functions::FEFieldFunction<dim>(dof_handler, global_solution));

  is_set_global_weights  = true;
  is_set_global_solution = true;
}


template <int dim, class ReconstructionType>
std::shared_ptr<Functions::FEFieldFunction<dim>>
SemiLagrangeBasis<dim, ReconstructionType>::
  get_local_field_function()
{
  Assert(
    is_set_global_solution,
    ExcMessage(
      "Cannot return local field function since global solution is not set."));

  return local_field_function_ptr;
}


template <int dim, class ReconstructionType>
void
SemiLagrangeBasis<dim, ReconstructionType>::initialize()
{
  Timer timer;
  timer.restart();

  make_grid();

  setup_system();

  if (verbose)
    {
      std::cout << "   Initializing local basis in global cell id  "
                << this->global_cell_id.to_string() << "   (subdomain "
                << this->local_subdomain << "     "
                << triangulation.n_active_cells() << " active fine cells     "
                << dof_handler.n_dofs() << " subgrid dofs) ....";
    }
  if (verbose_all)
    std::cout << std::endl;

  {
    /*
     * The global solution must be set in this case for an initial
     * reconstruction.
     */
    AffineConstraints<double> constraints_fake;
    constraints_fake.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints_fake);
    constraints_fake.close();

    VectorTools::project(dof_handler,
                         constraints_fake,
                         QGauss<dim>(fe.degree + 1),
                         initial_value,
                         global_solution);

    local_field_function_ptr.reset(
      new Functions::FEFieldFunction<dim>(dof_handler, global_solution));

    is_set_global_solution = true;
  }

  basis_reconstructor.initialize(this->global_cell,
                                 time_step,
                                 /* n_steps_local */ 1,
                                 matrix_coeff,
                                 advection_field,
                                 right_hand_side,
                                 solution_vector,
                                 solution_vector_old,
                                 solution_vector_time_derivative);

  /*
   * Must be set with appropriate name for this timestep before output.
   */
  set_filename_global();

  if (verbose)
    {
      timer.stop();
      std::cout << "	done in   " << timer.cpu_time() << "   seconds."
                << std::endl;
    }
}


template <int dim, class ReconstructionType>
void
SemiLagrangeBasis<dim, ReconstructionType>::initial_reconstruction()
{
  /*
   * Initial reconstruction for the basis can now be done since the initial
   * global solution is locally set.
   */
  time = 0.0;
  basis_reconstructor.basis_initial_reconstruction();

  if (this->is_first_cell)
    {
      output_basis(solution_vector_old);
    }
}


template <int dim, class ReconstructionType>
void
SemiLagrangeBasis<dim, ReconstructionType>::make_time_step()
{
  Timer timer;
  if (verbose)
    {
      timer.restart();
      std::cout << "   Time step for local basis in global cell id  "
                << this->global_cell_id.to_string() << "   (subdomain "
                << this->local_subdomain << "     "
                << triangulation.n_active_cells() << " active fine cells     "
                << dof_handler.n_dofs() << " subgrid dofs) ....";
    }
  if (verbose_all)
    std::cout << std::endl;

  time += time_step;
  ++timestep_number;

  // reset
  is_set_global_weights        = false;
  this->is_set_filename_global = false;
  is_solved                    = false;

  {
    basis_reconstructor.basis_reconstruction(/* current time = */ time,
                                             /* dt = */ time_step,
                                             /* theta = */ theta);
    is_solved = true;
  }

  basis_reconstructor.assemble_global_element_data(global_element_matrix,
                                                   global_element_matrix_old,
                                                   global_element_rhs,
                                                   global_element_rhs_old,
                                                   time_step,
                                                   theta);

  /*
   * Must be set with appropriate name for this timestep before output.
   */
  set_filename_global();

  if (output_first_basis && this->is_first_cell)
    output_basis(solution_vector);

  if (verbose)
    {
      timer.stop();

      std::cout << "	done in   " << timer.cpu_time() << "   seconds."
                << std::endl;
    }
}

}
#endif /* DIFFUSION_PROBLEM_INCLUDE_INVERSE_MATRIX_HPP_ */
