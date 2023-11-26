/*
 * advectiondiffusion_basis.hpp
 *
 *  Created on: Jan 10, 2020
 *      Author: heena
 */

#ifndef INCLUDE_ADVECTIONDIFFUSION_BASIS_HPP_
#define INCLUDE_ADVECTIONDIFFUSION_BASIS_HPP_

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
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

// STL
#include <stdexcept>
#include <vector>


// My Headers
#include "advection_field.hpp"
#include "config.h"
#include "dirichlet_bc.hpp"
#include "matrix_coeff.hpp"
#include "neumann_bc.hpp"
#include "q1basis.hpp"
#include "right_hand_side.hpp"
#include "basis_interface.hpp"
/*!
 * @namespace Timedependent_AdvectionDiffusionProblem
 * @brief Contains implementation of the main object
 * and all functions to solve a time-independent
 * Dirichlet-Neumann problem on a unit square.
 */
namespace Timedependent_AdvectionDiffusionProblem
{
using namespace dealii;

template <int dim>
class AdvectionDiffusionBasisFirst: public BasisInterface<dim>
{
public:
  /*
   * Constructor.
   */
  AdvectionDiffusionBasisFirst() = delete;

  /*!
   * Default constructor.
   */
  AdvectionDiffusionBasisFirst(typename Triangulation<dim>::active_cell_iterator &global_cell,
		  	  	  	  	  bool is_first_cell, unsigned int local_subdomain, const double theta,
						   MPI_Comm mpi_communicator,
						   AdvectionDiffusionBase<dim> &_global_problem);

  /*!      const FE_Q<dim> &                  _fe,
   * Copy constructor.
   */
  AdvectionDiffusionBasisFirst(const AdvectionDiffusionBasisFirst &X);

  ~AdvectionDiffusionBasisFirst();

   /*!
   * Initialization function of the object. Must be called before first time
   * step update.
   */
  void initialize()override;

  /*!
   * Make a global time step.
   */
  void make_time_step()override;

  /*!
   * Write out global solution in cell.
   */
  void output_global_solution_in_cell() const override;

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
  const Vector<double> &get_global_element_rhs(bool current_time_flag) const override;
  /*!
   * Return filename for local pvtu record.
   */
  /*!
   * @brief Set global weights.
   * @param weights
   *
   * The coarse weights of the global solution determine
   * the local multiscale solution. They must be computed
   * and then set locally to write an output.
   */
  void
      set_global_weights(const std::vector<double> &global_weights) override;

      /*!
       * Get an info string to append to filenames.
       */
      virtual const std::string
      get_basis_info_string() override;

private:


  /*!
   * @brief Set up the grid with a certain number of refinements.
   *
   * Generate a triangulation of \f$[0,1]^{\rm{dim}}\f$ with edges/faces
   * numbered form \f$1,\dots,2\rm{dim}\f$.
   */
  void make_grid();

  /*!
   * @brief Setup sparsity pattern and system matrix.
   *
   * Compute sparsity pattern and reserve memory for the sparse system matrix
   * and a number of right-hand side vectors. Also build a constraint object
   * to take care of Dirichlet boundary conditions.
   */
  void setup_system();
  /*!
   * @brief Assemble the global element rhs and matrix from local basis data.
   *
   * This is the routine whose details are determined by the global time
   * stepping algorithm and by the concrete problem at hand.
   */
  void assemble_global_element_data();
  /*!
   * @brief Assemble the system matrix and the static right hand side.
   *
   * Assembly routine to build the time-independent (static) part.
   * Neumann boundary conditions will be put on edges/faces
   * with odd number. Constraints are not applied here yet.
   */
  void assemble_system(double current_time);

  /*!
   * @brief Assemble the gloabl element matrix and the gobla right hand side.
   */
  void
  assemble_global_element_matrix(const SparseMatrix<double> &relevant_matrix,
                                 FullMatrix<double> &global_data_matrix,
                                 const double factor,
                                 const bool use_time_derivative_trial_function,
                                 const bool at_current_time_step);
  void assemble_global_element_rhs(const Vector<double> &local_forcing,
                                   Vector<double> &global_data_vector,
                                   const double factor,
                                   const bool at_current_time_step);

  /*!
   * @brief Sparse direct solver.
   *
   * Sparse direct solver through UMFPACK.
   */
  void solve_direct(unsigned int index_basis);
  /*!
   * @brief Iterative solver.
   *
   * Parallel sparse direct solver through Amesos package.
   */
  // void solve_direct (unsigned int index_basis);
  /*!
   * @brief Iterative solver.
   *
   * CG-based solver with preconditioning.
   */

  void solve_iterative(unsigned int index_basis);

  /*!
   * @brief Compute time derivative of basis at current time step.
   *
   * Compute time derivative of basis at current time step using backward
   * differencing.
   */
  void compute_time_derivative();

  /*!
   * @brief Write basis results to disk.
   *
   * Write basis results to disk in vtu-format.
   */
  void output_basis(const std::vector<Vector<double>> &solution_vector) const;


  /*!
   * Define the global filename for pvtu-file in global output.
   */
  void set_filename_global();

  const unsigned int n_refine_local = 4;
  const bool         verbose            = false;
   const bool         verbose_all        = false;
  const bool use_direct_solver = true;
  const bool output_first_basis = true;
  const std::string filename_global_base;


  Triangulation<dim> triangulation;

  FE_Q<dim> fe;
  DoFHandler<dim> dof_handler;

  std::vector<AffineConstraints<double>> constraints_vector;

  std::vector<Point<dim>> corner_points;

  SparsityPattern sparsity_pattern;

  /*!
   * Mass matrix without constraints.
   */
  SparseMatrix<double> mass_matrix;

  /*!
   * Advection matrix without constraints.
   */
  SparseMatrix<double> advection_matrix;

  /*!
   * Advection matrix without constraints at previous time.
   */
  SparseMatrix<double> advection_matrix_old;

  /*!
   * Diffusion matrix without constraints.
   */
  SparseMatrix<double> diffusion_matrix;

  /*!
   * Diffusion matrix without constraints.
   */
  SparseMatrix<double> diffusion_matrix_old;

  /*!
   * System matrix without constraints.
   */
  SparseMatrix<double> system_matrix;

  /*!
   * System matrix with constraints.
   */
  SparseMatrix<double> system_matrix_with_constraints;

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
  Vector<double> global_rhs;

  /*!
   * Contains the right-hand side.
   */
  Vector<double>
      global_rhs_old; // this is only for the global assembly (speed-up)

  /*!
   * Contains all parts of the right-hand side needed to
   * solve the linear system to make a step from k-1 to k
   * but this time with constraints. This variable only serves
   * as a temporary variable in make_time_step().
   */

  /*!
   * Contains all parts of the right-hand side needed to
   * solve the linear system..
   */
  Vector<double> system_rhs;
  /*!
   * This variable only serves
   * as a temporary variable in make_time_step().
   */
  Vector<double> tmp;

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
   * Weights of multiscale basis functions.
   */
  std::vector<double> global_weights;

  /*!
   * Bool to guard against writing global solution is when global weights are
   * uninitialized.
   */
  bool is_set_global_weights;

  /*!
   * Global solution
   */
  Vector<double> global_solution;


  /*!
   * Object carries set of local \f$Q_1\f$-basis functions.
   */
  Coefficients::Q1Basis<dim> q1basis;
  //  bool is_set_basis_data;
  /*!
   * Make sure basis object is initialized before time stepping.
   */
  bool is_initialized;

  double time;
  double time_step;
  unsigned int timestep_number;

  /*!
   * parameter to determine the "implicitness" of the method.
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


};

template <int dim>
AdvectionDiffusionBasisFirst<dim>::AdvectionDiffusionBasisFirst(typename Triangulation<dim>::active_cell_iterator &global_cell,
													  bool is_first_cell, unsigned int local_subdomain, const double theta,
													  MPI_Comm mpi_communicator,
													  AdvectionDiffusionBase<dim> &                      global_problem)
    :  BasisInterface<dim>(global_cell,
            is_first_cell,
            local_subdomain,
            mpi_communicator,
            global_problem),
	  filename_global_base((dim == 2 ? "solution-ms_2d" : "solution-ms_3d")),
      fe(1), dof_handler(triangulation),
      constraints_vector(GeometryInfo<dim>::vertices_per_cell),
      corner_points(GeometryInfo<dim>::vertices_per_cell),
      solution_vector(GeometryInfo<dim>::vertices_per_cell),
      solution_vector_old(GeometryInfo<dim>::vertices_per_cell),
      solution_vector_time_derivative(GeometryInfo<dim>::vertices_per_cell),
      is_solved(false),
      global_element_matrix(fe.dofs_per_cell, fe.dofs_per_cell),
      global_element_matrix_old(fe.dofs_per_cell, fe.dofs_per_cell),
      global_element_rhs(fe.dofs_per_cell),
      global_element_rhs_old(fe.dofs_per_cell),
      global_weights(fe.dofs_per_cell, 0), is_set_global_weights(false),
       q1basis(global_cell),
      is_initialized(false), time(0.0), time_step(1. / 100), timestep_number(0),
      /*
       * theta=1 is implicit Euler,
       * theta=0 is explicit Euler,
       * theta=0.5 is Crank-Nicolson
       */
      theta(theta)

{
  // set corner points
  for (unsigned int vertex_n = 0;vertex_n < GeometryInfo<dim>::vertices_per_cell; ++vertex_n)
  {
    corner_points[vertex_n] = global_cell->vertex(vertex_n);
  }
}

template <int dim>
AdvectionDiffusionBasisFirst<dim>::AdvectionDiffusionBasisFirst(const AdvectionDiffusionBasisFirst<dim> &X)
    : BasisInterface<dim>(X),
      n_refine_local(X.n_refine_local), verbose(X.verbose),
      verbose_all(X.verbose_all), use_direct_solver(X.use_direct_solver),
      output_first_basis(X.output_first_basis),
      filename_global_base(X.filename_global_base),
       triangulation(), fe(X.fe),
      dof_handler(triangulation), constraints_vector(X.constraints_vector),
      corner_points(X.corner_points), sparsity_pattern(X.sparsity_pattern),
      mass_matrix(X.mass_matrix), advection_matrix(X.advection_matrix),
      advection_matrix_old(X.advection_matrix_old),
      diffusion_matrix(X.diffusion_matrix),
      diffusion_matrix_old(X.diffusion_matrix_old),
      system_matrix(X.system_matrix),
      system_matrix_with_constraints(X.system_matrix_with_constraints),
      solution_vector(X.solution_vector),
      solution_vector_old(X.solution_vector_old),
      solution_vector_time_derivative(X.solution_vector_time_derivative),
      is_solved(X.is_solved), global_rhs(X.global_element_rhs),
      global_rhs_old(X.global_element_rhs_old), system_rhs(X.system_rhs),
      tmp(X.tmp), global_element_matrix(X.global_element_matrix),
      global_element_matrix_old(X.global_element_matrix),
      global_element_rhs(X.global_element_rhs),
      global_element_rhs_old(X.global_element_rhs_old),
      global_weights(X.global_weights),
      is_set_global_weights(X.is_set_global_weights),
      global_solution(X.global_solution),
      q1basis(X.q1basis), is_initialized(X.is_initialized), time(X.time),
      time_step(X.time_step), timestep_number(X.time_step),
      /*
       * theta=1 is implicit Euler,
       * theta=0 is explicit Euler,
       * theta=0.5 is Crank-Nicolson
       */
      theta(X.theta), matrix_coeff(X.matrix_coeff),
      advection_field(X.advection_field), right_hand_side(X.right_hand_side)
	  {}

template <int dim> AdvectionDiffusionBasisFirst<dim>::~AdvectionDiffusionBasisFirst()
{
  mass_matrix.clear();
  advection_matrix.clear();
  advection_matrix_old.clear();
  diffusion_matrix.clear();
  diffusion_matrix_old.clear();
  system_matrix.clear();
  system_matrix_with_constraints.clear();

  for (unsigned int index_basis = 0; index_basis < solution_vector.size(); ++index_basis)
  {
    constraints_vector[index_basis].clear();
  }
  dof_handler.clear();
}

template <int dim> void AdvectionDiffusionBasisFirst<dim>::make_grid()
{

  /*A general dim -dimensional cell (a segment if dim is 1, a quadrilateral if
   * dim is 2, or a hexahedron if dim is 3) immersed in a spacedim -dimensional
   * space. It is the responsibility of the user to provide the vertices in the
   * right order (see the documentation of the GeometryInfo class) because the
   * vertices are stored in the same order as they are given. It is also
   * important to make sure that the volume of the cell is positive.*/
  GridGenerator::general_cell(triangulation, corner_points, /* colorize faces */false);

  triangulation.refine_global(n_refine_local);

}

template <int dim> void AdvectionDiffusionBasisFirst<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);


  for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell; ++i)
  {

    q1basis.set_index(i);

    constraints_vector[i].clear();

    DoFTools::make_hanging_node_constraints(dof_handler, constraints_vector[i]);

    VectorTools::interpolate_boundary_values(dof_handler,
                                             /*boundary id*/ 0, q1basis,
                                             constraints_vector[i]);
    constraints_vector[i].close();
  }

  /*
   * Set up Dirichlet boundary conditions and sparsity pattern.
   */
  {
    DynamicSparsityPattern dsp(dof_handler.n_dofs());

    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints_vector[0],
                                    /*keep_constrained_dofs = */ true);

    sparsity_pattern.copy_from(dsp);
  }
  system_matrix.reinit(sparsity_pattern);
  system_matrix_with_constraints.reinit(sparsity_pattern);

  mass_matrix.reinit(sparsity_pattern);
  advection_matrix.reinit(sparsity_pattern);
  advection_matrix_old.reinit(sparsity_pattern);
  diffusion_matrix.reinit(sparsity_pattern);
  diffusion_matrix_old.reinit(sparsity_pattern);

  for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell; ++i)
  {
    solution_vector[i].reinit(dof_handler.n_dofs());
    solution_vector_old[i].reinit(dof_handler.n_dofs());
    solution_vector_time_derivative[i].reinit(dof_handler.n_dofs());
  }

  system_rhs.reinit(dof_handler.n_dofs());
  tmp.reinit(dof_handler.n_dofs());

  global_rhs.reinit(dof_handler.n_dofs());
  global_rhs_old.reinit(dof_handler.n_dofs());
}

template <int dim>
void AdvectionDiffusionBasisFirst<dim>::assemble_system(double current_time)
{
  const QGauss<dim> quadrature_formula(fe.degree + 1);

  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();

  FullMatrix<double> cell_matrix_mass(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_matrix_advection(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_matrix_advection_old(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_matrix_diffusion(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_matrix_diffusion_old(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);
  Vector<double> cell_rhs_old(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<Tensor<2, dim>> matrix_coeff_values_old(n_q_points);
  std::vector<Tensor<2, dim>> matrix_coeff_values(n_q_points);

  std::vector<double> rhs_values_old(n_q_points);
  std::vector<double> rhs_values(n_q_points);

  std::vector<Tensor<1, dim>> advection_field_values_old(n_q_points);
  std::vector<Tensor<1, dim>> advection_field_values(n_q_points);


  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    cell_matrix_mass = 0;
    cell_matrix_advection = 0;
    cell_matrix_advection_old = 0;
    cell_matrix_diffusion = 0;
    cell_matrix_diffusion_old = 0;
    cell_rhs = 0;
    cell_rhs_old = 0;

    fe_values.reinit(cell);
    cell->get_dof_indices(local_dof_indices);
    /*
     * Values at current time.
     */
    matrix_coeff.set_time(current_time);
    advection_field.set_time(current_time);
    right_hand_side.set_time(current_time);

    advection_field.value_list(fe_values.get_quadrature_points(),
                               advection_field_values);
    matrix_coeff.value_list(fe_values.get_quadrature_points(),
                            matrix_coeff_values);
    right_hand_side.value_list(fe_values.get_quadrature_points(), rhs_values);


    /*
     * Values at previous time.
     */
    matrix_coeff.set_time(current_time - time_step);
    advection_field.set_time(current_time - time_step);
    right_hand_side.set_time(current_time - time_step);

    advection_field.value_list(fe_values.get_quadrature_points(),
                               advection_field_values_old);
    matrix_coeff.value_list(fe_values.get_quadrature_points(),
                            matrix_coeff_values_old);
    right_hand_side.value_list(fe_values.get_quadrature_points(),
                               rhs_values_old);

    for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
    {
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          // Diffusion is on rhs. Careful with signs here.
          // on LHS
          cell_matrix_mass(i, j) += fe_values.shape_value(i, q_index) *
                                    fe_values.shape_value(j, q_index) *
                                    fe_values.JxW(q_index);
          // on LHS
          cell_matrix_advection(i, j) += fe_values.shape_value(i, q_index) *
                                         advection_field_values[q_index] *
                                         fe_values.shape_grad(j, q_index) *
                                         fe_values.JxW(q_index);
          // on LHS
          cell_matrix_advection_old(i, j) +=
              fe_values.shape_value(i, q_index) *
              advection_field_values_old[q_index] *
              fe_values.shape_grad(j, q_index) * fe_values.JxW(q_index);
          // on RHS (note the sign)
          cell_matrix_diffusion(i, j) -=
              fe_values.shape_grad(i, q_index) * matrix_coeff_values[q_index] *
              fe_values.shape_grad(j, q_index) * fe_values.JxW(q_index);
          // on RHS (note the sign)
          cell_matrix_diffusion_old(i, j) -= fe_values.shape_grad(i, q_index) *
                                             matrix_coeff_values_old[q_index] *
                                             fe_values.shape_grad(j, q_index)*
                                             fe_values.JxW(q_index);
        } // ++j
        // on RHS
        cell_rhs(i) += fe_values.shape_value(i, q_index) * rhs_values[q_index] *
                       fe_values.JxW(q_index);
        // on RHS
        cell_rhs_old(i) += fe_values.shape_value(i, q_index) *
                           rhs_values_old[q_index] * fe_values.JxW(q_index);
      } // end ++i
    }   // end ++q_index

    // get global indices
    cell->get_dof_indices(local_dof_indices);

    /*
     * Now add the cell matrix and rhs to the right spots
     * in the global matrix and global rhs. Constraints will
     * be taken care of later.
     */
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
    {
      for (unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        mass_matrix.add(local_dof_indices[i], local_dof_indices[j],
                        cell_matrix_mass(i, j));
        advection_matrix.add(local_dof_indices[i], local_dof_indices[j],
                             cell_matrix_advection(i, j));
        advection_matrix_old.add(local_dof_indices[i], local_dof_indices[j],
                                 cell_matrix_advection_old(i, j));
        diffusion_matrix.add(local_dof_indices[i], local_dof_indices[j],
                             cell_matrix_diffusion(i, j));
        diffusion_matrix_old.add(local_dof_indices[i], local_dof_indices[j],
                                 cell_matrix_diffusion_old(i, j));
      }
      global_rhs(local_dof_indices[i]) += cell_rhs(i);
      global_rhs_old(local_dof_indices[i]) += cell_rhs_old(i);
    }

  } // ++cell
}

template <int dim>
void AdvectionDiffusionBasisFirst<dim>::assemble_global_element_data()
{
  /*
   * We assemble the global system Mu_t + Cu = Au + f. In time discrete
   * form this reads (Mu)' - Nu + Cu = Au + f.
   * With the theta-method and (Nu)_i = <phi_i', phi_j>u_j which amounts to
   * [M^{n+1} + dt*theta*(C-N-A)]u^{n+1} = [M^n + dt*(1-theta)*(A+N-C)]u^n
   * + theta*f^{n+1} + (1-theta)*f^n
   */

  Assert(is_solved,
         ExcMessage("System for all basis functions must be solved before "
                    "global data can be assembled in each time step."));

  compute_time_derivative();

  // First reset
  global_element_matrix = 0;
  global_element_matrix_old = 0;
  global_element_rhs = 0;
  global_element_rhs_old = 0;

  {
    // Mass matrix
    assemble_global_element_matrix(
        mass_matrix, global_element_matrix,
        /* factor */ 1,
        /* use_time_derivative_test_function */ false,
        /* at_current_time_step */ true);

    // Mass matrix at previous time
    assemble_global_element_matrix(
        mass_matrix, global_element_matrix_old,
        /* factor */ 1,
        /* use_time_derivative_test_function */ false,
        /* at_current_time_step */ false);
  }

  if (!Timedependent_AdvectionDiffusionProblemUtilities::is_approx(theta,
                                                                   0.0)) {
    /*
     * Means we do not have a fully explicit method
     * so that we must assemble more than just mass
     * for the system matrix.
     */

    // Mass matrix derivative N
    assemble_global_element_matrix(mass_matrix, global_element_matrix,
                                   /* factor */ (-1) * theta * time_step,
                                   /* use_time_derivative_test_function */ true,
                                   /* at_current_time_step */ true);

    // Advection matrix C
    assemble_global_element_matrix(
        advection_matrix, global_element_matrix,
        /* factor */ theta * time_step,
        /* use_time_derivative_test_function */ false,
        /* at_current_time_step */ true);

    // Diffusion matrix D
    assemble_global_element_matrix(
        diffusion_matrix, global_element_matrix,
        /* factor */ (-1) * theta * time_step,
        /* use_time_derivative_test_function */ false,
        /* at_current_time_step */ true);

    // Forcing at current time step
    assemble_global_element_rhs(global_rhs, global_element_rhs, theta,
                                /* at_current_time_step */ true);
  }

  if (!Timedependent_AdvectionDiffusionProblemUtilities::is_approx(theta,
                                                                   1.0))
  {
    /*
     * Means we do not have a fully implicit method
     * so that we must assemble more than just mass
     * for the system matrix.
     */

    // Mass matrix derivative N at previous time step
    assemble_global_element_matrix(mass_matrix, global_element_matrix_old,
                                   /* factor */ (1 - theta) * time_step,
                                   /* use_time_derivative_test_function */ true,
                                   /* at_current_time_step */ false);

    // Advection matrix C at previous time step
    assemble_global_element_matrix(
        advection_matrix_old, global_element_matrix_old,
        /* factor */ (-1) * (1 - theta) * time_step,
        /* use_time_derivative_test_function */ false,
        /* at_current_time_step */ false);

    // Diffusion matrix A at previous time step
    assemble_global_element_matrix(
        diffusion_matrix_old, global_element_matrix_old,
        /* factor */ (1 - theta) * time_step,
        /* use_time_derivative_test_function */ false,
        /* at_current_time_step */ false);

    // Forcing at previous time step
    assemble_global_element_rhs(global_rhs_old, global_element_rhs_old,
                                (1 - theta),
                                /* at_current_time_step */ false);
  }
}
template <int dim>
void AdvectionDiffusionBasisFirst<dim>::assemble_global_element_matrix(const SparseMatrix<double> &relevant_matrix,
																  FullMatrix<double> &global_data_matrix, const double factor,
																  const bool use_time_derivative_test_function,
																  const bool at_current_time_step)
{
  // Get lengths of tmp vectors for assembly
  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  Vector<double> tmp(dof_handler.n_dofs());
  std::vector<Vector<double>> &relevant_test_vector =
      (at_current_time_step ? (use_time_derivative_test_function
                                   ? solution_vector_time_derivative
                                   : solution_vector)
                            : (use_time_derivative_test_function
                                   ? solution_vector_time_derivative
                                   : solution_vector_old));
  std::vector<Vector<double>> &relevant_trial_vector =
      (at_current_time_step ? solution_vector : solution_vector_old);

  // This assembles the local contribution to the global global matrix
  // with an algebraic trick. It uses the local system matrix stored in
  // the respective basis object.
  for (unsigned int i_test = 0; i_test < dofs_per_cell; ++i_test)
  {
    // set an alias name
    const Vector<double> &test_vec = relevant_test_vector[i_test];

    for (unsigned int i_trial = 0; i_trial < dofs_per_cell; ++i_trial)
    {
      // set an alias name
      const Vector<double> &trial_vec = relevant_trial_vector[i_trial];

      // tmp = system_matrix*trial_vec
      relevant_matrix.vmult(tmp, trial_vec);

      // global_element_diffusion_matrix = test_vec*tmp
      global_data_matrix(i_test, i_trial) += factor * (test_vec * tmp);

      // reset
      tmp = 0;
    } // end for i_trial
  }   // end for i_test
}

template <int dim>
void AdvectionDiffusionBasisFirst<dim>::assemble_global_element_rhs(const Vector<double> &local_forcing,
															   Vector<double> &global_data_vector,
															   const double factor, const bool at_current_time_step)
{
  // Get lengths of tmp vectors for assembly
  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  std::vector<Vector<double>> &relevant_test_vector =
      (at_current_time_step ? solution_vector : solution_vector_old);

  // This assembles the local contribution to the global global matrix
  // with an algebraic trick. It uses the local system matrix stored in
  // the respective basis object.
  for (unsigned int i_test = 0; i_test < dofs_per_cell; ++i_test) {
    // set an alias name
    const Vector<double> &test_vec = relevant_test_vector[i_test];

    global_data_vector(i_test) += test_vec * local_forcing;
    global_data_vector(i_test) *= factor;

  } // end for i_test
}

template <int dim>
void AdvectionDiffusionBasisFirst<dim>::compute_time_derivative()
{
  for (unsigned int index_basis = 0;index_basis < GeometryInfo<dim>::vertices_per_cell; ++index_basis)
  {
    solution_vector_time_derivative[index_basis] = 0;

    solution_vector_time_derivative[index_basis] +=
        solution_vector[index_basis];
    solution_vector_time_derivative[index_basis] -=
        solution_vector_old[index_basis];

    solution_vector_time_derivative[index_basis] /= time_step;
  }
}

template <int dim>
const FullMatrix<double> &
AdvectionDiffusionBasisFirst<dim>::get_global_element_matrix(bool current_time_flag) const
{
  if (current_time_flag)
  {
    return global_element_matrix;
  } else {

    return global_element_matrix_old;
  }
}

template <int dim>
const Vector<double> &AdvectionDiffusionBasisFirst<dim>::get_global_element_rhs(bool current_time_flag) const
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

template <int dim>
void AdvectionDiffusionBasisFirst<dim>::solve_direct(unsigned int index_basis)
{
  Timer timer;
  if (verbose_all)
  {
    std::cout << "	Solving linear system (SparseDirectUMFPACK) in cell   "
              << this->global_cell_id.to_string() << "   for basis   " << index_basis
              << ".....";

    timer.restart();
  }

  // use direct solver
  SparseDirectUMFPACK A_inv;
  A_inv.initialize(system_matrix_with_constraints);

  A_inv.vmult(solution_vector[index_basis], system_rhs);

  constraints_vector[index_basis].distribute(solution_vector[index_basis]);

  is_solved = true;

  if (verbose_all)
  {
    timer.stop();
    std::cout << "done in   " << timer.cpu_time() << "   seconds." << std::endl;
  }
}

template <int dim>
void AdvectionDiffusionBasisFirst<dim>::solve_iterative(unsigned int index_basis)
{
  Timer timer;
  if (verbose_all)
  {
	  std::cout << "	Solving linear system (CG with SSOR) in cell   "
	                 << this->global_cell_id.to_string() << "   for basis   "
	                 << index_basis << ".....";

    timer.restart();
  }

  SolverControl solver_control(1000, 1e-12);
  SolverCG<> solver(solver_control);

  PreconditionSSOR<> preconditioner;
  preconditioner.initialize(system_matrix, 1.6);

  solver.solve(system_matrix, solution_vector[index_basis], system_rhs,
               preconditioner);

  constraints_vector[index_basis].distribute(solution_vector[index_basis]);

  is_solved = true;

  if (verbose_all)
    std::cout << "   "
              << "done in   " << timer.cpu_time() << "   seconds after   "
              << solver_control.last_step() << "   CG iterations." << std::endl;
}

template <int dim>
const std::string
AdvectionDiffusionBasisFirst<dim>::get_basis_info_string()
{
  return "_AdvectionDiffusionBasisFirst_basis";
}


template <int dim>
void AdvectionDiffusionBasisFirst<dim>::output_basis( const std::vector<Vector<double>> &solution_vector) const
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

  for (unsigned int index_basis = 0;index_basis < GeometryInfo<dim>::vertices_per_cell; ++index_basis)
  {
    std::vector<std::string> solution_names(
        1, "u_" + Utilities::int_to_string(index_basis, 1));
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        interpretation(1, DataComponentInterpretation::component_is_scalar);

    data_out.add_data_vector(solution_vector[index_basis], solution_names,
                             DataOut<dim>::type_dof_data, interpretation);
  }

  data_out.build_patches();

  // filename
  std::string filename = "basis_q_static";
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
    std::cout << "done in   " << timer.cpu_time() << "   seconds." << std::endl;
  }
}

template <int dim>
void AdvectionDiffusionBasisFirst<dim>::set_global_weights(const std::vector<double> &weights)
{
  // Copy assignment of global weights
  global_weights = weights;

  // reinitialize the global solution on this cell
  global_solution.reinit(dof_handler.n_dofs());

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  // Set global solution using the weights and the local basis.
  for (unsigned int index_basis = 0; index_basis < dofs_per_cell;
       ++index_basis) {
    // global_solution = 1*global_solution +
    // global_weights[index_basis]*solution_vector[index_basis]
    global_solution.sadd(1, global_weights[index_basis],
                         solution_vector[index_basis]);
  }

  is_set_global_weights = true;
}

template <int dim>
void AdvectionDiffusionBasisFirst<dim>::output_global_solution_in_cell() const
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

template <int dim>
void
AdvectionDiffusionBasisFirst<dim>::set_filename_global()
{
  this->filename_global = filename_global_base + get_basis_info_string() + "." +
                          Utilities::int_to_string(this->local_subdomain, 5) +
                          ".cell-" + this->global_cell_id.to_string() +
                          ".theta-" + Utilities::to_string(theta, 4) +
                          ".time_step-" +
                          Utilities::int_to_string(timestep_number, 4) + ".vtu";

  this->is_set_filename_global = true;
}




template <int dim> void AdvectionDiffusionBasisFirst<dim>::initialize()
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

  // initialize with initial condition
  for (unsigned int index_basis = 0;index_basis < GeometryInfo<dim>::vertices_per_cell; ++index_basis)
  {
    q1basis.set_index(index_basis);

    VectorTools::project(dof_handler, constraints_vector[index_basis],
                         QGauss<dim>(fe.degree + 1), q1basis,
                         solution_vector_old[index_basis]);

    solution_vector[index_basis] = solution_vector_old[index_basis];
  }

  /*
   * Must be set with appropriate name for this timestep before output.
   */
  set_filename_global();

  if (this->is_first_cell)
    output_basis(solution_vector_old);

  is_initialized = true;

  if (verbose)
  {
    timer.stop();
    std::cout << "	done in   " << timer.cpu_time() << "   seconds."
              << std::endl;
  }
}

template <int dim> void AdvectionDiffusionBasisFirst<dim>::make_time_step()
{
	Assert(is_initialized, ExcNotInitialized());
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

  assemble_system(time);

  { // build the linear system to solve at this time step
    system_matrix = 0;

    /*
     * We assemble the local system Mu' + Cu = Au + f. In time discrete
     * (with the theta-method) this amounts to
     * [M + dt*theta*(C-A)]u^{n+1} = [M + dt*(1-theta)*(A-C)]u^n
     * + theta*f^{n+1} + (1-theta)*f^n
     */
    system_matrix.copy_from(mass_matrix);
    system_matrix.add(-theta * time_step, diffusion_matrix);
    system_matrix.add(theta * time_step, advection_matrix);
  }

  // reset
  is_set_global_weights = false;
  this->is_set_filename_global = false;
  is_solved = false;

  // Now solve for each basis (meaning with different constraints)
  for (unsigned int index_basis = 0; index_basis < GeometryInfo<dim>::vertices_per_cell; ++index_basis)
  {
    /*
     * The constraints are different for each basis and we
     * do not want to build the system matrix with constraints
     * every time again.
     */
    system_matrix_with_constraints = 0;
    system_rhs = 0;
    tmp = 0;

    /*
     * system_rhs = M*old_solution + (1-theta)*dt*(A-C)*old_solution
     *
     * Note that there is no forcing.
     */
    mass_matrix.vmult(system_rhs, solution_vector_old[index_basis]);
    diffusion_matrix.vmult(tmp, solution_vector_old[index_basis]);
    system_rhs.add((1 - theta) * time_step, tmp);
    tmp = 0;
    advection_matrix.vmult(tmp, solution_vector_old[index_basis]);
    system_rhs.add(-(1 - theta) * time_step, tmp);

    system_matrix_with_constraints.copy_from(system_matrix);

    // Now take care of constraints
    constraints_vector[index_basis].condense(system_matrix_with_constraints,
                                             system_rhs);

    // Now solve
    if (dim == 2 && use_direct_solver)
    {
      solve_direct(index_basis);
    }
    else
    {
      solve_iterative(index_basis);
    }

    solution_vector_old[index_basis] = solution_vector[index_basis];
  }

  assemble_global_element_data();

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

} // namespace Timedependent_AdvectionDiffusionProblem

#endif /* INCLUDE_ADVECTIONDIFFUSION_BASIS_HPP_ */
