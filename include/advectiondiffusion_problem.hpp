/*
 * diffusion_problem.hpp
 *
 *  Created on: Oct 7, 2019
 *      Author: heena
 */

#ifndef INCLUDE_AdvectionDiffusionProblem_PROBLEM_HPP_
#define INCLUDE_AdvectionDiffusionProblem_PROBLEM_HPP_

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

#include <deal.II/grid/grid_generator.h>
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

// STL
#include <cmath>
#include <fstream>
#include <iostream> // std::cout, std::endl

// My Headers
#include "advection_field.hpp"
#include "config.h"
#include "dirichlet_bc.hpp"
#include "initial_value.hpp"
#include "matrix_coeff.hpp"
#include "neumann_bc.hpp"
#include "right_hand_side.hpp"

/*!
 * Contains implementation of the main object
 * and all functions to solve a time dependent
 * Dirichlet-Neumann problem on a unit square.
 *
 * @namespace Timedependent_AdvectionDiffusionProblemProblem
 * @author Heena, Patel 2019
 */
namespace Timedependent_AdvectionDiffusionProblem {
using namespace dealii;

/*!
 * This class solves an advection-diffusion
 * problem in 2d or 3d with either periodic
 * or mixed Dirichlet-Neumann conditions.
 *
 * @author Heena, Patel 2019
 */
template <int dim> class AdvectionDiffusionProblem {
public:
  /*!
   * Standard constructor disabled.
   */
  AdvectionDiffusionProblem() = delete;

  /*!
   * Default constructor.
   */
  AdvectionDiffusionProblem(unsigned int n_refine, bool is_periodic);

  /*!
    * Destructor.
    */
   ~AdvectionDiffusionProblem();

  /*!
   * @brief Run function of the object.
   *
   * Run the computation after object is built. Implements theping loop.
   */
  void run();

private:
  /*!
   * @brief Set up the grid with a certain number of refinements
   * with either peridic or non-periodic bounday conditions.
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
   * @brief Assemble the system matrix and the right hand side at currtent time.
   *
   * Assembly routine to build the time-dependent matrix and rhs.
   * Neumann boundary conditions will be put on edges/faces
   * with odd number. Constraints are applied here.
   */
  void assemble_system (double current_time);

  /*!
   * @brief Iterative solver.
   *
   * Parallel sparse direct solver through Amesos package.
   */
  void solve_direct();

  /*!
   * @brief Iterative solver.
   *
   * CG-based solver with preconditioning.
   */
  void solve_iterative();

  /*!
   * @brief Write results to disk.
   *
   * Write results to disk in vtu-format.
   */
  void output_results(TrilinosWrappers::MPI::Vector& vector_out) const;

  MPI_Comm mpi_communicator;

  /*!
   * Distributed triangulation
   */
  parallel::distributed::Triangulation<dim> triangulation;

  FE_Q<dim> fe;
  DoFHandler<dim> dof_handler;

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
   * Time-dependent scalar coefficient (boundary flux).
   */
  Coefficients::NeumannBC<dim> neumann_bc;

  /*!
   * Index Set
   */
  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  AffineConstraints<double> constraints;

  TrilinosWrappers::SparsityPattern sparsity_pattern;

  TrilinosWrappers::SparseMatrix system_matrix;

  TrilinosWrappers::MPI::Vector solution;
  TrilinosWrappers::MPI::Vector old_solution;
  TrilinosWrappers::MPI::Vector system_rhs;

  ConditionalOStream pcout;
  TimerOutput computing_timer;

  double time;
  double time_step;
  unsigned int timestep_number;

  /*!
   * parameter to determine the "implicitness" of the method.
   * Zero is fully implicit and one is (almost explicit).
   */
  const double theta;

  /*!
   * Final simulation time.
   */
  const double T_max;

  /*!
   * Number of initial refinements.
   */
  unsigned int n_refine;

  /*!
   * If this flag is true then periodic boundary conditions
   * are used.
   */
  bool is_periodic;
};

template <int dim>
AdvectionDiffusionProblem<dim>::AdvectionDiffusionProblem(unsigned int n_refine,
                                            bool         is_periodic)
  : mpi_communicator(MPI_COMM_WORLD)
  , triangulation(mpi_communicator,
                  typename Triangulation<dim>::MeshSmoothing(
                    Triangulation<dim>::smoothing_on_refinement |
                    Triangulation<dim>::smoothing_on_coarsening))
  , fe(1)
  , dof_handler(triangulation)
  , pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
  , computing_timer(mpi_communicator,
                    pcout,
                    TimerOutput::summary,
                    TimerOutput::wall_times)
  , time(0.0)
  , time_step(1. / 100)
  , timestep_number(0)
  /*
   * theta=1 is implicit Euler,
   * theta=0 is explicit Euler,
   * theta=0.5 is Crank-Nicolson
   */
  , theta(1.0)
  , T_max(0.5)
  , n_refine(n_refine)
  , is_periodic(is_periodic)
{}


template <int dim>
AdvectionDiffusionProblem<dim>::~AdvectionDiffusionProblem()
{
  system_matrix.clear();
  constraints.clear();
  dof_handler.clear();
}


template <int dim>
void
AdvectionDiffusionProblem<dim>::make_grid()
{
  TimerOutput::Scope t(computing_timer, "mesh generation");

//  GridGenerator::hyper_rectangle(triangulation,
//		  	  	  	  	  	  	  	  	  	  Point<2>(-300.0, 0.0),
//		                                     Point<2>(500.0, 120.0),
//                                     true);

  GridGenerator::hyper_cube(triangulation, 0,1,true);

  if (is_periodic)
    {
      std::vector<
        GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
        periodicity_vector;

      for (unsigned int d = 0; d < dim; ++d)
        {
          GridTools::collect_periodic_faces(triangulation,
                                            /*b_id1*/ 2 * (d + 1) - 2,
                                            /*b_id2*/ 2 * (d + 1) - 1,
                                            /*direction*/ d,
                                            periodicity_vector);
        }

      triangulation.add_periodicity(periodicity_vector);
    } // if

  triangulation.refine_global(n_refine);
}


template <int dim>
void
AdvectionDiffusionProblem<dim>::setup_system()
{
  TimerOutput::Scope t(computing_timer, "system setup");

  dof_handler.distribute_dofs(fe);

  locally_owned_dofs = dof_handler.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);


  solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

  constraints.clear();
  constraints.reinit(locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);

  if (is_periodic)
    {
      std::vector<
        GridTools::PeriodicFacePair<typename DoFHandler<dim>::cell_iterator>>
        periodicity_vector;

      for (unsigned int d = 0; d < dim; ++d)
        {
          GridTools::collect_periodic_faces(dof_handler,
                                            /*b_id1*/ 2 * (d + 1) - 2,
                                            /*b_id2*/ 2 * (d + 1) - 1,
                                            /*direction*/ d,
                                            periodicity_vector);
        }

      DoFTools::make_periodicity_constraints<DoFHandler<dim>>(
        periodicity_vector, constraints);
    } // if
  else
    {
	    const Coefficients::DirichletBC<dim> dirichlet_bc;
      /*
       * Set up Dirichlet boundary conditions.
       */
      for (unsigned int i = 0; i < dim; ++i)
        {
          VectorTools::interpolate_boundary_values(dof_handler,
                                                   /*boundary id*/
                                                     2*i, // only even boundary id
                                                   dirichlet_bc,
                                                   constraints);
        }
    } // else

  constraints.close();

  // Now initialize the spasity pattern
  sparsity_pattern.reinit(locally_owned_dofs, mpi_communicator);
  DoFTools::make_sparsity_pattern(dof_handler,
                                  sparsity_pattern,
                                  constraints,
                                  /* keep_constrained_dofs */ true,
                                  Utilities::MPI::this_mpi_process(
                                    mpi_communicator));
  sparsity_pattern.compress();

  system_matrix.reinit(sparsity_pattern);

  solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

  old_solution.reinit(locally_owned_dofs,
                      locally_relevant_dofs,
                      mpi_communicator);
  system_rhs.reinit(locally_owned_dofs, mpi_communicator);
}


template <int dim>
void
AdvectionDiffusionProblem<dim>::assemble_system(double current_time)
{
  TimerOutput::Scope t(computing_timer, "assembly");

  const QGauss<dim>     quadrature_formula(fe.degree + 1);
  const QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);

  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  FEFaceValues<dim> fe_face_values(
    fe,
    face_quadrature_formula,
    update_values | update_quadrature_points | update_normal_vectors |
      update_JxW_values); // for Neumaan boundary condition to evaluate
                          // boundary condition

  const unsigned int dofs_per_cell   = fe.dofs_per_cell;
  const unsigned int n_q_points      = quadrature_formula.size();
  const unsigned int n_face_q_points = face_quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<Tensor<2, dim>> matrix_coeff_values_old(n_q_points);
  std::vector<Tensor<2, dim>> matrix_coeff_values(n_q_points);

  std::vector<Tensor<1, dim>> advection_field_values_old(n_q_points);
  std::vector<Tensor<1, dim>> advection_field_values(n_q_points);

  std::vector<double> rhs_values_old(n_q_points);
  std::vector<double> rhs_values(n_q_points);

  std::vector<double> neumann_values_old(n_face_q_points);
  std::vector<double> neumann_values(n_face_q_points);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          cell_matrix = 0;
          cell_rhs    = 0;

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
          right_hand_side.value_list(fe_values.get_quadrature_points(),
                                     rhs_values);

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

          /*
           * Integration over cell.
           */
          for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
            {
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      // Diffusion is on rhs. Careful with signs here.
                      cell_matrix(i, j) +=
                        (fe_values.shape_value(i, q_index) *
                           fe_values.shape_value(j, q_index) +
                         time_step * (theta) *
                           (fe_values.shape_grad(i, q_index) *
                              matrix_coeff_values[q_index] *
                              fe_values.shape_grad(j, q_index) +
                            fe_values.shape_value(i, q_index) *
                              advection_field_values[q_index] *
                              fe_values.shape_grad(j, q_index))) *
                        fe_values.JxW(q_index);
                      // Careful with signs also here.
                      cell_rhs(i) += (fe_values.shape_value(i, q_index) *
                                        fe_values.shape_value(j, q_index) -
                                      time_step * (1 - theta) *
                                        (fe_values.shape_grad(i, q_index) *
                                           matrix_coeff_values_old[q_index] *
                                           fe_values.shape_grad(j, q_index) +
                                         fe_values.shape_value(i, q_index) *
                                           advection_field_values_old[q_index] *
                                           fe_values.shape_grad(j, q_index))) *
                                     fe_values.JxW(q_index) *
                                     old_solution(local_dof_indices[j]);
                    } // end ++j

                  cell_rhs(i) += time_step * fe_values.shape_value(i, q_index) *
                                 ((1 - theta) * rhs_values_old[q_index] +
                                  (theta)*rhs_values[q_index]) *
                                 fe_values.JxW(q_index);
                } // end ++i
            }     // end ++q_index

          if (!is_periodic)
            {
              /*
               * Boundary integral for Neumann values for odd boundary_id in
               * non-periodic case.
               */
              for (unsigned int face_number = 0;
                   face_number < GeometryInfo<dim>::faces_per_cell;
                   ++face_number)
                {
                  if (cell->face(face_number)->at_boundary() &&
                      ((cell->face(face_number)->boundary_id() ==1) ||
                       (cell->face(face_number)->boundary_id() == 3) ||
                       (cell->face(face_number)->boundary_id() == 5)))
                    {
                      fe_face_values.reinit(cell, face_number);

                      /*
                       * Fill in values at this particular face at current
                       * time.
                       */
                      neumann_bc.set_time(current_time);
                      neumann_bc.value_list(
                        fe_face_values.get_quadrature_points(), neumann_values);

                      /*
                       * Fill in values at this particular face at previous
                       * time.
                       */
                      neumann_bc.set_time(current_time - time_step);
                      neumann_bc.value_list(
                        fe_face_values.get_quadrature_points(),
                        neumann_values_old);

                      for (unsigned int q_face_point = 0;
                           q_face_point < n_face_q_points;
                           ++q_face_point)
                        {
                          for (unsigned int i = 0; i < dofs_per_cell; ++i)
                            {
                              cell_rhs(i) +=
                                time_step *
                                ((1 - theta) *
                                 neumann_values_old
                                   [q_face_point] * // g(x_q, t_n) = A*grad_u
                                                    // at t_n
                                 +(theta)*neumann_values
                                   [q_face_point]) * // g(x_q, t_{n+1}) =  =
                                                     // A*grad_u at t_{n+1}
                                fe_face_values.shape_value(
                                  i, q_face_point) *              // phi_i(x_q)
                                fe_face_values.JxW(q_face_point); // dS
                            }                                     // end ++i
                        } // end ++q_face_point
                    }     // end if
                }         // end ++face_number
            }

          constraints.distribute_local_to_global(
            cell_matrix,
            cell_rhs,
            local_dof_indices,
            system_matrix,
            system_rhs,
            /* use_inhomogeneities_for_rhs */ true);
        } // if
    }     // ++cell

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
}


template <int dim>
void
AdvectionDiffusionProblem<dim>::solve_direct()
{
  TimerOutput::Scope t(computing_timer,
                       "parallel sparse direct solver (MUMPS)");

  TrilinosWrappers::MPI::Vector completely_distributed_solution(
    locally_owned_dofs, mpi_communicator);

  SolverControl                  solver_control;
  TrilinosWrappers::SolverDirect solver(solver_control);
  solver.initialize(system_matrix);

  solver.solve(system_matrix, completely_distributed_solution, system_rhs);

  pcout << "   Solved in with direct solver." << std::endl;

  constraints.distribute(completely_distributed_solution);

  solution = completely_distributed_solution;
}


template <int dim>
void
AdvectionDiffusionProblem<dim>::solve_iterative()
{
  TimerOutput::Scope t(computing_timer, "iterative solver");

  TrilinosWrappers::MPI::Vector completely_distributed_solution(
    locally_owned_dofs, mpi_communicator);

  SolverControl solver_control(
    /* max number of iterations */ dof_handler.n_dofs(),
    /* tolerance */ 1e-12,
    /* print log history */ true);

  TrilinosWrappers::SolverGMRES cg_solver(solver_control);

  //  TrilinosWrappers::PreconditionAMG preconditioner;
  //  TrilinosWrappers::PreconditionAMG::AdditionalData data;
  TrilinosWrappers::PreconditionIdentity                 preconditioner;
  TrilinosWrappers::PreconditionIdentity::AdditionalData data;
  //  TrilinosWrappers::PreconditionSSOR preconditioner;
  //  TrilinosWrappers::PreconditionSSOR::AdditionalData data(/* omega = */ 1,
  //		/* min_diagonal  = */ 0,
  //		/* overlap = */ 0,
  //		/* n_sweeps = */ 1 );

  preconditioner.initialize(system_matrix, data);

  cg_solver.solve(system_matrix,
                  completely_distributed_solution,
                  system_rhs,
                  preconditioner);

  pcout << "   Solved in " << solver_control.last_step()
        << " iterations (theta = " << theta << ")." << std::endl;

  constraints.distribute(completely_distributed_solution);

  solution = completely_distributed_solution;
}

template <int dim>
void
AdvectionDiffusionProblem<dim>::output_results(
  TrilinosWrappers::MPI::Vector &vector_out) const
{
  std::string  filename = (dim == 2 ? "solution-std_2d" : "solution-std_3d");
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(vector_out, "u");

  Vector<float> subdomain(triangulation.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    {
      subdomain(i) = triangulation.locally_owned_subdomain();
    }
  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches();

  std::string filename_slave(filename);
  filename_slave +=
    "_refinements-" + Utilities::int_to_string(n_refine, 1) + "." + "theta-" +
    Utilities::to_string(theta, 4) + "." + "time_step-" +
    Utilities::int_to_string(timestep_number, 4) + "." +
    Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4) +
    ".vtu";

  std::ofstream output(filename_slave.c_str());
  data_out.write_vtu(output);

  /*
   * Write a pvtu-record to collect all files for each time step.
   */
  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      std::vector<std::string> file_list;
      for (unsigned int i = 0;
           i < Utilities::MPI::n_mpi_processes(mpi_communicator);
           ++i)
        {
          file_list.push_back(
            filename + "_refinements-" + Utilities::int_to_string(n_refine, 1) +
            "." + "theta-" + Utilities::to_string(theta, 4) + "." +
            "time_step-" + Utilities::int_to_string(timestep_number, 4) + "." +
            Utilities::int_to_string(i, 4) + ".vtu");
        }

      std::string filename_master(filename);

      filename_master +=
        "_refinements-" + Utilities::int_to_string(n_refine, 1) + "." +
        "theta-" + Utilities::to_string(theta, 4) + "." + "time_step-" +
        Utilities::int_to_string(timestep_number, 4) + ".pvtu";

      std::ofstream master_output(filename_master);
      data_out.write_pvtu_record(master_output, file_list);
    }
}

template <int dim>
void
AdvectionDiffusionProblem<dim>::run()
{
  pcout << std::endl
        << "===========================================" << std::endl;

  pcout << "Running " << dim << "D Problem (with Trilinos) on "
        << Utilities::MPI::n_mpi_processes(mpi_communicator)
        << " MPI rank(s)..." << std::endl;


  make_grid();

  setup_system();

  pcout << "   Number of active cells:       "
        << triangulation.n_global_active_cells() << std::endl
        << "   Number of degrees of freedom: " << dof_handler.n_dofs()
        << std::endl;

  // initialize with initial condition
  TrilinosWrappers::MPI::Vector tmp(locally_owned_dofs, mpi_communicator);
  VectorTools::project(
    dof_handler, constraints, QGauss<dim>(fe.degree + 1), Coefficients::InitialValue<dim>(), tmp);

  // output initial condition
  old_solution = tmp;
  output_results(old_solution);

  /*
   * In each time step we need to build the system to be solved.
   * If theta=1 the method is fully implicit. If theta=0 the method
   * is explicit in the sense that we only have to solve a system
   * involving a mass matrix.
   * EVENTUALLY WE SOLVE system_matrix*solution=system_rhs.
   */
  while (time <= T_max)
    {
      time += time_step;
      ++timestep_number;

      pcout << "Time step " << timestep_number << " at t=" << time << std::endl;

      assemble_system(time);

      // Now solve
      if (dim == 2 && dof_handler.n_dofs() < std::pow(2.5 * 10, 6))
        {
          solve_direct();
        }
      else
        {
          solve_iterative();
        }

      {
        TimerOutput::Scope t(computing_timer, "output vtu");
        output_results(solution);
      }

      /*
       * Hand over solution and forcing values for next time step to
       * avoid unnecessary re-assembly.
       */
      old_solution = solution;

      /*
       * Reinitialize the system data.
       */
      system_matrix.reinit(sparsity_pattern);
      system_rhs.reinit(locally_owned_dofs, mpi_communicator);

      if ((timestep_number > 0) && (timestep_number % 10 == 0))
        {
          pcout << std::endl
                << "------------   Wall clock after   " << timestep_number
                << "   time steps   ------------" << std::endl;
          computing_timer.print_summary();
          pcout
            << "----------------------------------------------------------------------------------------"
            << std::endl;
        }
    } // while

  pcout << std::endl
        << "===========================================" << std::endl;

  computing_timer.print_summary();
  computing_timer.reset();
}



} // namespace Timedependent_AdvectionDiffusionProblemProblem


#endif /* INCLUDE_DIFFUSION_PROBLEM_HPP_ */
