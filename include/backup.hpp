/*
 * backup.hpp
 *
 *  Created on: Jun 10, 2020
 *      Author: heena
 */

#ifndef INCLUDE_BACKUP_HPP_
#define INCLUDE_BACKUP_HPP_

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
#include "advectiondiffusion_basis_reconstruction.hpp"
#include "config.h"
#include "dirichlet_bc.hpp"
#include "initial_value.hpp"
#include "matrix_coeff.hpp"
#include "neumann_bc.hpp"
#include "right_hand_side.hpp"

namespace Timedependent_AdvectionDiffusionProblem
{
using namespace dealii;

template <int dim>
class AdvectionDiffusionProblemMultiscale : public ParameterAcceptor
{
public:
  AdvectionDiffusionProblemMultiscale() = delete;
  AdvectionDiffusionProblemMultiscale(unsigned int n_refine, bool is_periodic);
  ~AdvectionDiffusionProblemMultiscale();
  void run();

private:
  void make_grid();
  void setup_system();
  void initialize_and_compute_basis();
  // void setup_constraints();
  void assemble_system(double current_time); /* Needs to be modified. */
  void solve_direct();
  void solve_iterative();
  void send_global_weights_to_cell(
      const TrilinosWrappers::MPI::Vector &some_solution);
  std::vector<std::string> collect_filenames_on_mpi_process() const;
  void output_results(TrilinosWrappers::MPI::Vector &vector_out) const;

  MPI_Comm mpi_communicator;

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
   * Dirichlet conditions. They can be time-dependent but must match the initial
   * condition.
   */
  Coefficients::DirichletBC<dim> dirichlet_bc;

  /*!
   * Initial conditions
   */
  Coefficients::InitialValue<dim> initial_value;
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
  /**
   * Contains all parts of the right-hand side needed to
   * solve the linear system.
   */

  TrilinosWrappers::MPI::Vector system_rhs;

  using BasisType =
      Timedependent_AdvectionDiffusionProblem::AdvectionDiffusionBasis<dim>;

  // using Reconstruction =
  // Timedependent_AdvectionDiffusionProblem::AdvectionDiffusionBasisReconstruction<dim>;

  /*!
   * STL Vector holding basis functions for each coarse cell.
   */
  std::map<CellId, BasisType> cell_basis_map;

  // std::map<CellId,Reconstruction> cell_basis_reconstruction_map;

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
   * Number of global refinements.
   */
  const unsigned int n_refine;

  /*!
   * If this flag is true then periodic boundary conditions
   * are used.
   */
  bool is_periodic;
};

template <int dim>
AdvectionDiffusionProblemMultiscale<dim>::AdvectionDiffusionProblemMultiscale(
    unsigned int n_refine, bool is_periodic)
    : mpi_communicator(MPI_COMM_WORLD),
      triangulation(mpi_communicator,
                    typename Triangulation<dim>::MeshSmoothing(
                        Triangulation<dim>::smoothing_on_refinement |
                        Triangulation<dim>::smoothing_on_coarsening)),
      fe(1), dof_handler(triangulation),
      pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
      computing_timer(mpi_communicator, pcout, TimerOutput::summary,
                      TimerOutput::wall_times),
      time(0.0), time_step(1. / 100), timestep_number(0),
      /*
       * theta=1 is implicit Euler,
       * theta=0 is explicit Euler,
       * theta=0.5 is Crank-Nicolson
       */
      theta(0.0), T_max(1.0), n_refine(n_refine), is_periodic(is_periodic)

{}

template <int dim>
AdvectionDiffusionProblemMultiscale<
    dim>::~AdvectionDiffusionProblemMultiscale()
{
  system_matrix.clear();
  constraints.clear();
  dof_handler.clear();
}

template <int dim> void AdvectionDiffusionProblemMultiscale<dim>::make_grid()
{
  TimerOutput::Scope t(computing_timer, "coarse mesh generation");

  GridGenerator::hyper_cube(triangulation, 0, 1,
                            /* colorize */ true);

  if (is_periodic)
  {
    std::vector<GridTools::PeriodicFacePair<
        typename parallel::distributed::Triangulation<dim>::cell_iterator>>
        periodicity_vector;

    for (unsigned int d = 0; d < dim; ++d)
    {
      GridTools::collect_periodic_faces(triangulation,
                                        /*b_id1*/ 2 * (d + 1) - 2,
                                        /*b_id2*/ 2 * (d + 1) - 1,
                                        /*direction*/ d, periodicity_vector);
    }

    triangulation.add_periodicity(periodicity_vector);
  } // if

  triangulation.refine_global(n_refine);
}

template <int dim>
void AdvectionDiffusionProblemMultiscale<dim>::setup_system()
{
  TimerOutput::Scope t(computing_timer, "system  setup");

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
                                        /*direction*/ d, periodicity_vector);
    }

    DoFTools::make_periodicity_constraints<DoFHandler<dim>>(periodicity_vector,
                                                            constraints);
  } // if
  else {
    /*
     * Set up Dirichlet boundary conditions.
     */
    const Coefficients::DirichletBC<dim> dirichlet_bc;
    for (unsigned int i = 0; i < dim; ++i)
    {
      VectorTools::interpolate_boundary_values(dof_handler,
                                               /*boundary id*/ 2 *
                                                   i, // only even boundary id
                                               dirichlet_bc, constraints);
    }
  } // else

  constraints.close();

  // Now initialize the spasity pattern
  sparsity_pattern.reinit(locally_owned_dofs, mpi_communicator);
  DoFTools::make_sparsity_pattern(dof_handler, sparsity_pattern, constraints,
      /* keep_constrained_dofs */ true,
      Utilities::MPI::this_mpi_process(mpi_communicator));
  sparsity_pattern.compress();

  system_matrix.reinit(sparsity_pattern);

  solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

  old_solution.reinit(locally_owned_dofs, locally_relevant_dofs,
                      mpi_communicator);
  system_rhs.reinit(locally_owned_dofs, mpi_communicator);
}

template <int dim>
void AdvectionDiffusionProblemMultiscale<dim>::initialize_and_compute_basis()

{
  TimerOutput::Scope t(computing_timer, "basis initialization and computation");

  typename Triangulation<dim>::active_cell_iterator cell = dof_handler
                                                               .begin_active(),
                                                    endc = dof_handler.end();
  const CellId first_cell = cell->id(); // only to identify first cell (for
                                        // setting output flag for basis)

  for (; cell != endc; ++cell)
  {
    if (cell->is_locally_owned())
    {
      const bool is_first_cell = (first_cell == cell->id());
      BasisType current_cell_problem(cell, is_first_cell,
                                     triangulation.locally_owned_subdomain(),
                                     theta, mpi_communicator);
      const CellId current_cell_id(cell->id());

      std::pair<typename std::map<CellId, BasisType>::iterator, bool> result;
      result = cell_basis_map.insert(
          std::make_pair(current_cell_id, current_cell_problem));

      Assert(result.second,ExcMessage("Insertion of local basis problem into std::map failed. "
                     "Problem with copy constructor?"));
    }
  } // end ++cell

  typename std::map<CellId, BasisType>::iterator it_basis =
                                                     cell_basis_map.begin(),
                                                 it_endbasis =
                                                     cell_basis_map.end();
  for (; it_basis != it_endbasis; ++it_basis)
  {
    (it_basis->second).initialize();
  }
}

template <int dim>
void AdvectionDiffusionProblemMultiscale<dim>::assemble_system(double current_time)
{
  TimerOutput::Scope t(computing_timer, "assembly");

  const QGauss<dim> quadrature_formula(fe.degree + 1);
  const QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);

  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  FEFaceValues<dim> fe_face_values(
      fe, face_quadrature_formula,
      update_values | update_quadrature_points | update_normal_vectors |
          update_JxW_values); // for Neumaan boundary condition to evaluate
                              // boundary condition

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();
  const unsigned int n_face_q_points = face_quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_matrix_old(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<double> neumann_values_old(n_face_q_points);
  std::vector<double> neumann_values(n_face_q_points);

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    if (cell->is_locally_owned())
    {
      typename std::map<CellId, BasisType>::iterator it_basis =cell_basis_map.find(cell->id());

      cell_matrix = 0;
      cell_matrix_old = 0;
      cell_rhs = 0;

      fe_values.reinit(cell);
      cell->get_dof_indices(local_dof_indices);

      cell_matrix = (it_basis->second).get_global_element_matrix(true);
      cell_matrix_old = (it_basis->second).get_global_element_matrix(false);
      cell_rhs = (it_basis->second).get_global_element_rhs(true);
      cell_rhs += (it_basis->second).get_global_element_rhs(false);

      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        for (unsigned int j = 0; j < dofs_per_cell; ++j) {
          // Need additional contribution.
          cell_rhs(i) +=
              cell_matrix_old(i, j) * old_solution(local_dof_indices[j]);
        } // end ++j
      }   // end ++i

      if (!is_periodic)
      {
        /*
         * Boundary integral for Neumann values for odd boundary_id in
         * non-periodic case.
         */
        for (unsigned int face_number = 0;
             face_number < GeometryInfo<dim>::faces_per_cell; ++face_number)
        {
          if (cell->face(face_number)->at_boundary() &&
              ((cell->face(face_number)->boundary_id() == 1) ||
               (cell->face(face_number)->boundary_id() == 3) ||
               (cell->face(face_number)->boundary_id() == 5)))
          {
            fe_face_values.reinit(cell, face_number);

            /*
             * Fill in values at this particular face at current
             * time.
             */
            neumann_bc.set_time(current_time);
            neumann_bc.value_list(fe_face_values.get_quadrature_points(),
                                  neumann_values);

            /*
             * Fill in values at this particular face at previous
             * time.
             */
            neumann_bc.set_time(current_time - time_step);
            neumann_bc.value_list(fe_face_values.get_quadrature_points(),
                                  neumann_values_old);

            for (unsigned int q_face_point = 0; q_face_point < n_face_q_points;
                 ++q_face_point) {
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                cell_rhs(i) +=
                    time_step *
                    ((1 - theta) *
                     neumann_values_old[q_face_point] * // g(x_q, t_n) =
                                                        // A*grad_u at t_n
                     +(theta)*neumann_values[q_face_point]) * // g(x_q, t_{n+1})
                                                              // =  = A*grad_u
                                                              // at t_{n+1}
                    fe_face_values.shape_value(i, q_face_point) * // phi_i(x_q)
                    fe_face_values.JxW(q_face_point);             // dS
              }                                                   // end ++i
            } // end ++q_face_point
          }   // end if
        }     // end ++face_number
      }

      constraints.distribute_local_to_global(
          cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs,
          /* use_inhomogeneities_for_rhs */ true);
    }
  }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
}

template <int dim>
void AdvectionDiffusionProblemMultiscale<dim>::solve_direct()
{
  TimerOutput::Scope t(computing_timer,
                       "parallel sparse direct solver (MUMPS)");

  TrilinosWrappers::MPI::Vector completely_distributed_solution(
      locally_owned_dofs, mpi_communicator);

  SolverControl solver_control;
  TrilinosWrappers::SolverDirect solver(solver_control);
  solver.initialize(system_matrix);

  solver.solve(system_matrix, completely_distributed_solution, system_rhs);

  pcout << "   Solved in with direct solver." << std::endl;

  constraints.distribute(completely_distributed_solution);
  solution = completely_distributed_solution;
}

template <int dim>
void AdvectionDiffusionProblemMultiscale<dim>::solve_iterative()
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
  TrilinosWrappers::PreconditionIdentity preconditioner;
  TrilinosWrappers::PreconditionIdentity::AdditionalData data;
  //  TrilinosWrappers::PreconditionSSOR preconditioner;
  //  TrilinosWrappers::PreconditionSSOR::AdditionalData data(/* omega = */ 1,
  //		/* min_diagonal  = */ 0,
  //		/* overlap = */ 0,
  //		/* n_sweeps = */ 1 );

  preconditioner.initialize(system_matrix, data);

  cg_solver.solve(system_matrix, completely_distributed_solution, system_rhs,
                  preconditioner);

  pcout << "   Global problem solved in " << solver_control.last_step()
        << " iterations.(theta = " << theta << ")." << std::endl;

  constraints.distribute(completely_distributed_solution);
  solution = completely_distributed_solution;
}

template <int dim>
void AdvectionDiffusionProblemMultiscale<dim>::send_global_weights_to_cell(const TrilinosWrappers::MPI::Vector &some_solution)
{
  // For each cell we get dofs_per_cell values
  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // active cell iterator
  typename DoFHandler<dim>::active_cell_iterator cell =
                                                     dof_handler.begin_active(),
                                                 endc = dof_handler.end();
  for (; cell != endc; ++cell)
  {
    if (cell->is_locally_owned())
    {
      cell->get_dof_indices(local_dof_indices);
      std::vector<double> extracted_weights(dofs_per_cell, 0);
      some_solution.extract_subvector_to(local_dof_indices, extracted_weights);

      typename std::map<CellId, BasisType>::iterator it_basis =
          cell_basis_map.find(cell->id());
      (it_basis->second).set_global_weights(extracted_weights);
    }
  } // end ++cell
}

template <int dim>
std::vector<std::string>
AdvectionDiffusionProblemMultiscale<dim>::collect_filenames_on_mpi_process() const
{
  std::vector<std::string> filename_list;

  typename std::map<CellId, BasisType>::const_iterator it_basis = cell_basis_map.begin(),
                                                       it_endbasis = cell_basis_map.end();

  for (; it_basis != it_endbasis; ++it_basis)
  {
    filename_list.push_back((it_basis->second).get_filename_global());
  }

  return filename_list;
}

template <int dim>
void AdvectionDiffusionProblemMultiscale<dim>::output_results( TrilinosWrappers::MPI::Vector &vector_out) const
{
  // write local fine solution
  typename std::map<CellId, BasisType>::const_iterator it_basis = cell_basis_map.begin(),
                                                       it_endbasis = cell_basis_map.end();

  for (; it_basis != it_endbasis; ++it_basis)
  {
    (it_basis->second).output_global_solution_in_cell();
  }

  // Gather local filenames
  std::vector<std::string> filenames_on_cell;
  {
    std::vector<std::vector<std::string>> filename_list_list =
        Utilities::MPI::gather(mpi_communicator,
                               collect_filenames_on_mpi_process(),
                               /* root_process = */ 0);

    for (unsigned int i = 0; i < filename_list_list.size(); ++i)
      for (unsigned int j = 0; j < filename_list_list[i].size(); ++j)
        filenames_on_cell.emplace_back(filename_list_list[i][j]);
  }

  std::string filename = (dim == 2 ? "solution-ms_2d" : "solution-ms_3d");
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(vector_out, "u");

  Vector<float> subdomain(triangulation.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
  {
    subdomain(i) = triangulation.locally_owned_subdomain();
  }
  data_out.add_data_vector(subdomain, "subdomain");

  // Postprocess
  //      std::unique_ptr<Q_PostProcessor> postprocessor(
  //        new Q_PostProcessor(parameter_filename));
  //      data_out.add_data_vector(locally_relevant_solution, *postprocessor);

  data_out.build_patches();

  std::string filename_local_coarse(filename);
  filename_local_coarse +=
      "_coarse_refinements-" + Utilities::int_to_string(n_refine, 2) + "." +
      "theta-" + Utilities::to_string(theta, 4) + "." + "time_step-" +
      Utilities::int_to_string(timestep_number, 4) + "." +
      Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4) +
      ".vtu";

  std::ofstream output(filename_local_coarse.c_str());
  data_out.write_vtu(output);

  /*
   * Write a pvtu-record to collect all files for each time step.
   */
  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
  {
    std::vector<std::string> all_local_filenames_coarse;
    for (unsigned int i = 0;
         i < Utilities::MPI::n_mpi_processes(mpi_communicator); ++i)
    {
      all_local_filenames_coarse.push_back(
          filename + "_coarse_refinements-" +
          Utilities::int_to_string(n_refine, 2) + "." + "theta-" +
          Utilities::to_string(theta, 4) + "." + "time_step-" +
          Utilities::int_to_string(timestep_number, 4) + "." +
          Utilities::int_to_string(i, 4) + ".vtu");
    }

    std::string filename_master(filename);

    filename_master += "_coarse_refinements-" +
                       Utilities::int_to_string(n_refine, 2) + "." + "theta-" +
                       Utilities::to_string(theta, 4) + "." + "time_step-" +
                       Utilities::int_to_string(timestep_number, 4) + ".pvtu";

    std::ofstream master_output(filename_master);
    data_out.write_pvtu_record(master_output, all_local_filenames_coarse);
  }

  // pvtu-record for all local fine outputs
  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
  {
    std::string filename_master = filename;
    filename_master += "_fine_refinements-" +
                       Utilities::int_to_string(n_refine, 2) + "." + "theta-" +
                       Utilities::to_string(theta, 4) + "." + "time_step-" +
                       Utilities::int_to_string(timestep_number, 4) + ".pvtu";

    std::ofstream master_output(filename_master);
    data_out.write_pvtu_record(master_output, filenames_on_cell);
  }
}

template <int dim> void AdvectionDiffusionProblemMultiscale<dim>::run()
{
  pcout << std::endl
        << "===========================================" << std::endl;

  pcout << "Running (with Trilinos) on "
        << Utilities::MPI::n_mpi_processes(mpi_communicator)
        << " MPI rank(s)..." << std::endl;

  make_grid();

  setup_system();

  pcout << "   Number of active cells:       "
        << triangulation.n_global_active_cells() << std::endl
        << "   Number of degrees of freedom: " << dof_handler.n_dofs()
        << std::endl;

  initialize_and_compute_basis();

  { // initialize with initial condition
    // Vector must not have ghosted elements to be writable
    TrilinosWrappers::MPI::Vector tmp;
    tmp.reinit(locally_owned_dofs, mpi_communicator);

    VectorTools::project(dof_handler, constraints, QGauss<dim>(fe.degree + 1),
                         initial_value, tmp);
    old_solution = tmp;
  }

  // Send global weights of initial condition to cell.
  send_global_weights_to_cell(old_solution);

  output_results(old_solution);

  while (time <= T_max)
  {
    time += time_step;
    ++timestep_number;

    pcout << "Time step " << timestep_number << " at t=" << time << std::endl;

    {
      /*
       * Now each node possesses a set of basis objects. Initialize it
       * We need to compute them on each node and do so in
       * a locally threaded way.
       */
      typename std::map<CellId, BasisType>::iterator it_basis =
                                                         cell_basis_map.begin(),
                                                     it_endbasis =
                                                         cell_basis_map.end();
      for (; it_basis != it_endbasis; ++it_basis)
      {
        (it_basis->second).make_time_step();
      }
    }

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

    send_global_weights_to_cell(old_solution);

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
      pcout << "---------------------------------------------------------------"
               "-------------------------"
            << std::endl;
    }
  }

  pcout << std::endl
        << "===========================================" << std::endl;
  computing_timer.print_summary();
  computing_timer.reset();
}

} // namespace Timedependent_AdvectionDiffusionProblem

#endif /* INCLUDE_BACKUP_HPP_ */
