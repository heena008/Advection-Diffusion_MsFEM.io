/*
 * H1_conformal.hpp
 *
 *  Created on: Jan 1, 2022
 *      Author: heena
 */

#ifndef INCLUDE_H1_CONFORMAL_HPP_
#define INCLUDE_H1_CONFORMAL_HPP_

#include "config.h"
#include "q1basis.hpp"
#include "matrix_coeff.hpp"
#include "right_hand_side.hpp"
#include "neumann_bc.hpp"
#include "dirichlet_bc.hpp"
#include "advection_field.hpp"
#include "Multiscale_FEFieldFunction.hpp"
namespace Timedependent_AdvectionDiffusionProblem {
using namespace dealii;

template<int dim>
class AdvectionDiffusionBasis_Reconstruction {
public:
	/*
	 * Default Constructor.
	 */
	AdvectionDiffusionBasis_Reconstruction() = delete;

	 /*!
	       * Constructor. Does not initialize fully.
	       */

	AdvectionDiffusionBasis_Reconstruction(BasisInterface<dim> &        _local_basis,
	        MPI_Comm                           mpi_communicator,
	        const FE_Q<dim> &                  _fe,
	        Coefficients::MatrixCoeff<dim> &   _matrix_coeff,
	        Coefficients::AdvectionField<dim> &_advection_field,
	        Coefficients::RightHandSide<dim> & _right_hand_side,
	        std::vector<Vector<double>> &      _solution_vector,
	        std::vector<Vector<double>> &      _solution_vector_old,
	        std::vector<Vector<double>> &      _solution_vector_time_derivative,
	        bool                               _is_first_cell);
	   /*!
	       * Copy constructor is deleted.
	       */
	AdvectionDiffusionBasis_Reconstruction(const AdvectionDiffusionBasis_Reconstruction<dim> &other) = delete;

	      /*!
	       * Copy assignment is deleted.
	       */
	AdvectionDiffusionBasis_Reconstruction<dim> &
	      operator=(const AdvectionDiffusionBasis_Reconstruction<dim> &other) = delete;

	      /*!
	       * Destructor must be virtual.
	       */


    virtual ~AdvectionDiffusionBasis_Reconstruction()=0;
    /*!
      * Late initialization.
      */

	void
	initialize(Coefficients::MatrixCoeff<dim> &   _matrix_coeff,
            Coefficients::AdvectionField<dim> &_advection_field,
            Coefficients::RightHandSide<dim> & _right_hand_side,
            std::vector<Vector<double>> &      _solution_vector,
            std::vector<Vector<double>> &      _solution_vector_old,
            std::vector<Vector<double>> &_solution_vector_time_derivative);

	  /*!
	       * Virtual function to implement a
	       * reconstruction for the initial basis. This is done from an initial
	       * condition. Does nothing by default.
	       */

	 virtual void
	      basis_initial_reconstruction(){};
	   /*!
	       * Pure virtual function to implement a reconstruction step for the basis
	       * at current_time.
	       */

	      virtual void
	      basis_reconstruction(const double current_time,
	                           const double time_step,
	                           const double theta) = 0;
	      /*!
	           * Get an info string to append to filenames.
	           */

	      virtual const std::string
	      get_info_string() = 0;
	      /*!
	          * Interface to copy a triangulation into the object.
	          */

	      void
	      copy_triangulation(Triangulation<dim> &other_tria);
	      /*!
	          * Plot trace back mesh with info for diagnostic purposes. Only
	          * implemented in base class.
	          */

	      virtual void
	      print_traced_mesh_info(const Vector<double> &solution,
	                             const std::string &   filename) final;


	      /*
	       * These members and functions should be available in derived classes.
	       */

	    protected:

	      void
	      trace_back_mesh(const double current_time);

	      Coefficients::MatrixCoeff<dim> &
	      matrix_coeff();

	      Coefficients::AdvectionField<dim> &
	      advection_field();

	      Coefficients::RightHandSide<dim> &
	      right_hand_side();

	      std::vector<Vector<double>> &
	      solution_vector();

	      std::vector<Vector<double>> &
	      solution_vector_old();

	      std::vector<Vector<double>> &
	      solution_vector_time_derivative();

	      MPI_Comm mpi_communicator;

	      Triangulation<dim> local_tria;

	      DoFHandler<dim> dof_handler;

	      const FE_Q<dim> fe;

	    private:

	      Coefficients::MatrixCoeff<dim> *matrix_coeff_ptr;

	      /*!
	       * Reference to time-dependent vector coefficient (velocity).
	       */
	      Coefficients::AdvectionField<dim> *advection_field_ptr;

	      /*!
	       * Reference to time-dependent scalar coefficient (forcing).
	       */
	      Coefficients::RightHandSide<dim> *right_hand_side_ptr;


	      std::vector<Vector<double>> *solution_vector_ptr;


	      std::vector<Vector<double>> *solution_vector_old_ptr;

	      std::vector<Vector<double>> *solution_vector_time_derivative_ptr;

	      bool is_initialized;

	    protected:

	      bool is_first_cell;

	    private:

	      Coefficients::SemiLagrangian<dim> mesh_back_tracer;

	      const unsigned int n_refine_local = 4;

	    protected:
	     BasisInterface<dim> *local_basis_ptr;
	    };


template <int dim>
AdvectionDiffusionBasis_Reconstruction<dim>::AdvectionDiffusionBasis_Reconstruction(
			BasisInterface<dim> &        _local_basis,
	  MPI_Comm                           _mpi_communicator,
	  const FE_Q<dim> &                  _fe,
	  Coefficients::MatrixCoeff<dim> &   _matrix_coeff,
	  Coefficients::AdvectionField<dim> &_advection_field,
	  Coefficients::RightHandSide<dim> & _right_hand_side,
	  std::vector<Vector<double>> &      _solution_vector,
	  std::vector<Vector<double>> &      _solution_vector_old,
	  std::vector<Vector<double>> &      _solution_vector_time_derivative,
	  bool                               _is_first_cell)
	  : mpi_communicator(_mpi_communicator)
	  , local_tria()
	  , dof_handler(local_tria)
	  , fe(_fe)
	  , matrix_coeff_ptr(&_matrix_coeff)
	  , advection_field_ptr(&_advection_field)
	  , right_hand_side_ptr(&_right_hand_side)
	  , solution_vector_ptr(&_solution_vector)
	  , solution_vector_old_ptr(&_solution_vector_old)
	  , solution_vector_time_derivative_ptr(&_solution_vector_time_derivative)
	  , is_initialized(false)
	  , is_first_cell(_is_first_cell)
	  , mesh_back_tracer(_advection_field)
	  , local_basis_ptr(&_local_basis)
	{}


	template <int dim>
	AdvectionDiffusionBasis_Reconstruction<dim>::~AdvectionDiffusionBasis_Reconstruction()
	{}


	template <int dim>
	void
	AdvectionDiffusionBasis_Reconstruction<dim>::copy_triangulation(
	  Triangulation<dim> &other_tria)
	{

	  local_tria.copy_triangulation(other_tria);
	}


	template <int dim>
	void
	AdvectionDiffusionBasis_Reconstruction<dim>::initialize(
	  Coefficients::MatrixCoeff<dim> &   _matrix_coeff,
	  Coefficients::AdvectionField<dim> &_advection_field,
	  Coefficients::RightHandSide<dim> & _right_hand_side,
	  std::vector<Vector<double>> &      _solution_vector,
	  std::vector<Vector<double>> &      _solution_vector_old,
	  std::vector<Vector<double>> &      _solution_vector_time_derivative)
	{
	  local_tria.refine_global(n_refine_local);

	  dof_handler.distribute_dofs(fe);

	  matrix_coeff_ptr    = &_matrix_coeff;
	  advection_field_ptr = &_advection_field;
	  right_hand_side_ptr = &_right_hand_side;

	  solution_vector_ptr                 = &_solution_vector;
	  solution_vector_old_ptr             = &_solution_vector_old;
	  solution_vector_time_derivative_ptr = &_solution_vector_time_derivative;


	  is_initialized = true;
	}


	template <int dim>
	void
	AdvectionDiffusionBasis_Reconstruction<dim>::trace_back_mesh(
	  const double current_time)
	{
	  Assert(is_initialized, ExcNotInitialized());
	    {
	      mesh_back_tracer.set_time(current_time);
	      GridTools::transform(mesh_back_tracer, local_tria);

	    }
	}


	template <int dim>
	void
	AdvectionDiffusionBasis_Reconstruction<
	  dim>::print_traced_mesh_info(const Vector<double> &solution,
	                               const std::string &   filename)
	{
	  std::cout << "*** Writing traced back mesh ***" << std::endl
	            << "*** Mesh info:" << std::endl
	            << "*** Dimension: " << dim << std::endl
	            << "*** No. of cells: " << local_tria.n_active_cells() << std::endl;

	  {
	    std::map<types::boundary_id, unsigned int> boundary_count;
	    for (auto &cell : local_tria.active_cell_iterators())
	      {
	        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
	             ++face)
	          {
	            if (cell->face(face)->at_boundary())
	              boundary_count[cell->face(face)->boundary_id()]++;
	          }
	      }

	    std::cout << "*** Boundary indicators: ";
	    for (const std::pair<const types::boundary_id, unsigned int> &pair :
	         boundary_count)
	      {
	        std::cout << pair.first << "(" << pair.second << " times) ";
	      }
	    std::cout << std::endl;
	  }

	  DataOut<dim> data_out;
	  data_out.attach_dof_handler(dof_handler);

	  std::vector<std::string> solution_names(1, "u_traced_back");

	  std::vector<DataComponentInterpretation::DataComponentInterpretation>
	    data_component_interpretation(
	      1, DataComponentInterpretation::component_is_scalar);

	  data_out.add_data_vector(solution,
	                           solution_names,
	                           DataOut<dim>::type_dof_data,
	                           data_component_interpretation);

	  data_out.build_patches();

	  std::ofstream out_file_stream(filename);
	  data_out.write_vtu(out_file_stream);

	  std::cout << "*** Written to " << filename << std::endl;
	}


	template <int dim>
	std::vector<Vector<double>> &
	AdvectionDiffusionBasis_Reconstruction<dim>::solution_vector()
	{
	  Assert(is_initialized, ExcNotInitialized());

	  return *solution_vector_ptr;
	}


	template <int dim>
	std::vector<Vector<double>> &
	AdvectionDiffusionBasis_Reconstruction<dim>::solution_vector_old()
	{
	  Assert(is_initialized, ExcNotInitialized());

	  return *solution_vector_old_ptr;
	}


	template <int dim>
	std::vector<Vector<double>> &
	AdvectionDiffusionBasis_Reconstruction<
	  dim>::solution_vector_time_derivative()
	{
	  Assert(is_initialized, ExcNotInitialized());

	  return *solution_vector_time_derivative_ptr;
	}


	template <int dim>
	Coefficients::MatrixCoeff<dim> &
	AdvectionDiffusionBasis_Reconstruction<dim>::matrix_coeff()
	{
	  Assert(is_initialized, ExcNotInitialized());

	  return *matrix_coeff_ptr;
	}


	template <int dim>
	Coefficients::AdvectionField<dim> &
	AdvectionDiffusionBasis_Reconstruction<dim>::advection_field()
	{
	  Assert(is_initialized, ExcNotInitialized());

	  return *advection_field_ptr;
	}


	template <int dim>
	Coefficients::RightHandSide<dim> &
	AdvectionDiffusionBasis_Reconstruction<dim>::right_hand_side()
	{
	  Assert(is_initialized, ExcNotInitialized());

	  return *right_hand_side_ptr;
	}
template <int dim>
   class H1Conformal : public AdvectionDiffusionBasis_Reconstruction<dim>
   {
   public:
     /*!
      * Delete standard constructor.
      */
     H1Conformal() = delete;

     /*!
      * Constructor.
      */
     H1Conformal(
       BasisInterface<dim> &        _local_basis,
       MPI_Comm                           _mpi_communicator,
       FE_Q<dim> &                        _fe,
       Coefficients::MatrixCoeff<dim> &   _matrix_coeff,
       Coefficients::AdvectionField<dim> &_advection_field,
       Coefficients::RightHandSide<dim> & _right_hand_side,
       std::vector<Vector<double>> &      _solution_vector,
       std::vector<Vector<double>> &      _solution_vector_old,
       std::vector<Vector<double>> &      _solution_vector_time_derivative,
       bool                               _is_first_cell);

     /*!
      * Copy constructor is deleted.
      */
     H1Conformal(const H1Conformal<dim> &other) =
       delete;

     /*!
      * Copy assignment is deleted.
      */
     H1Conformal<dim> &
     operator=(const H1Conformal<dim> &other) = delete;

     /*!
      * Destructor.
      */
     virtual ~H1Conformal() override;

     /*!
      * Late initialization.
      */
     void
     initialize(
       const typename Triangulation<dim>::active_cell_iterator &global_cell,
       const double                                             _time_step,
       const unsigned int                                       _n_steps_local,
       Coefficients::MatrixCoeff<dim> &                         _matrix_coeff,
       Coefficients::AdvectionField<dim> &_advection_field,
       Coefficients::RightHandSide<dim> & _right_hand_side,
       std::vector<Vector<double>> &      _solution_vector,
       std::vector<Vector<double>> &      _solution_vector_old,
       std::vector<Vector<double>> &      _solution_vector_time_derivative);

     /*!
      * Implementation of \f$H^1\f$-conformal basis reconstruction basis
      * reconstruction for the initial basis. This is done from an initial
      * condition.
      */
     virtual void
     basis_initial_reconstruction() override;

     /*!
      * Implementation of \f$H^1\f$-conformal basis reconstruction basis
      * reconstruction.
      */
     virtual void
     basis_reconstruction(const double current_time,
                          const double time_step,
                          const double theta) override;

     /*!
      * @brief Assemble the global element rhs and matrix from local basis data.
      *
      * This is the routine whose details are determined by the global time
      * stepping algorithm and by the concrete problem at hand.
      */
     void
     assemble_global_element_data(
       FullMatrix<double> &global_element_matrix,
       FullMatrix<double> &global_element_matrix_old,
       Vector<double> &    global_element_rhs,
       Vector<double> &    global_element_matrix_rhs_old,
       const double        time_step,
       const double        theta);

     /*!
      * Get an info string to append to filenames.
      */
     virtual const std::string
     get_info_string() override;

   private:

     void
     setup_system();

     void
     project_reconstructed_function(double current_time, double time_step);

     void
     assemble_system(const double current_time, const double time_step);

     void
     assemble_global_element_matrix(
       const SparseMatrix<double> &relevant_matrix,
       FullMatrix<double> &        global_data_matrix,
       const double                factor,
       const bool                  use_time_derivative_trial_function,
       const bool                  at_current_time_step);

     /*!
      * @brief Assemble the global element rhs from local basis data.
      */
     void
     assemble_global_element_rhs(const Vector<double> &local_forcing,
                                 Vector<double> &      global_data_vector,
                                 const double          factor,
                                 const bool            at_current_time_step);

     /*!
      * @brief Compute time derivative of basis at current time step.
      *
      * Compute time derivative of basis at current time step using backward
      * differencing.
      */
     void
     compute_time_derivative(const double time_step);

     /*!
      * Object carries set of local \f$Q_1\f$-basis functions.
      */
     Coefficients::Q1Basis<dim> basis_q1;

     bool is_initialized;

     bool rebuild_system_matrix;

     AffineConstraints<double> constraints_fake;

     std::vector<AffineConstraints<double>> constraints_vector;
     SparsityPattern                        sparsity_pattern;

     /*!
      * Mass matrix without constraints.
      */
     SparseMatrix<double> mass_matrix;
     TrilinosWrappers::SparseMatrix system_matrix;
     /*!
      * Advection matrix without constraints.
      */
     SparseMatrix<double> advection_matrix;


     SparseMatrix<double> advection_matrix_old;

     SparseMatrix<double> diffusion_matrix;

     SparseMatrix<double> diffusion_matrix_old;

     Vector<double>
       global_rhs; // this is only for the global assembly (speed-up)

     Vector<double>
       global_rhs_old; // this is only for the global assembly (speed-up)

     Vector<double> reconstructed_solution;

     MsFEFieldFunctionMPI<dim> local_to_global_field_function;
   };

template <int dim>
H1Conformal<dim>::
  H1Conformal(
		  BasisInterface<dim> &        _local_basis,
    MPI_Comm                           _mpi_communicator,
    FE_Q<dim> &                        _fe,
    Coefficients::MatrixCoeff<dim> &   _matrix_coeff,
    Coefficients::AdvectionField<dim> &_advection_field,
    Coefficients::RightHandSide<dim> & _right_hand_side,
    std::vector<Vector<double>> &      _solution_vector,
    std::vector<Vector<double>> &      _solution_vector_old,
    std::vector<Vector<double>> &      _solution_vector_time_derivative,
    bool                               _is_first_cell)
  : AdvectionDiffusionBasis_Reconstruction<dim>(
      _local_basis,
      _mpi_communicator,
      _fe,
      _matrix_coeff,
      _advection_field,
      _right_hand_side,
      _solution_vector,
      _solution_vector_old,
      _solution_vector_time_derivative,
      _is_first_cell)
  , basis_q1()
  , is_initialized(false)
  , rebuild_system_matrix(true)
  , constraints_vector(GeometryInfo<dim>::vertices_per_cell)
  , local_to_global_field_function(_mpi_communicator, _local_basis)
{}


template <int dim>
H1Conformal<dim>::~H1Conformal()
{
  mass_matrix.clear();
  advection_matrix.clear();
  advection_matrix_old.clear();
  diffusion_matrix.clear();
  diffusion_matrix_old.clear();

  //  system_matrix.clear();
  //  system_matrix_with_constraints.clear();

  for (unsigned int index_basis = 0;
       index_basis < GeometryInfo<dim>::vertices_per_cell;
       ++index_basis)
    {
      constraints_vector[index_basis].clear();
    }
}


template <int dim>
void
H1Conformal<dim>::initialize(
  const typename Triangulation<dim>::active_cell_iterator &_global_cell,
  const double                                             _time_step,
  const unsigned int                                       _n_steps_local,
  Coefficients::MatrixCoeff<dim> &                         _matrix_coeff,
  Coefficients::AdvectionField<dim> &                      _advection_field,
  Coefficients::RightHandSide<dim> &                       _right_hand_side,
  std::vector<Vector<double>> &                            _solution_vector,
  std::vector<Vector<double>> &                            _solution_vector_old,
  std::vector<Vector<double>> &_solution_vector_time_derivative)
{
 AdvectionDiffusionBasis_Reconstruction<dim>::initialize(_time_step,
                                      _n_steps_local,
                                      _matrix_coeff,
                                      _advection_field,
                                      _right_hand_side,
                                      _solution_vector,
                                      _solution_vector_old,
                                      _solution_vector_time_derivative);

  basis_q1.initialize(_global_cell);

  setup_system();

  is_initialized = true;
}


template <int dim>
void H1Conformal<dim>::setup_system()
{
  /*
   * Set up Dirichlet boundary conditions and sparsity pattern.
   */

  // This is only for projecting the reconstruction of the deformed mesh.
  // Nothing special to be done here.
  constraints_fake.clear();
  DoFTools::make_hanging_node_constraints(this->dof_handler, constraints_fake);
  constraints_fake.close();

  // initialize with initial condition
  for (unsigned int index_basis = 0;
       index_basis < GeometryInfo<dim>::vertices_per_cell;
       ++index_basis)
    {
      basis_q1.set_index(index_basis);

      constraints_vector[index_basis].clear();
      DoFTools::make_hanging_node_constraints(this->dof_handler,
                                              constraints_vector[index_basis]);

      constraints_vector[index_basis].close();

      /*
       * This call must come after closing the constraints object.
       *
       */
      VectorTools::project(this->dof_handler,
                           constraints_vector[index_basis],
                           QGauss<dim>((this->fe).degree + 1),
                           basis_q1,
                           (this->solution_vector_old())[index_basis]);

      // Copy assignment
      (this->solution_vector())[index_basis] =
        (this->solution_vector_old())[index_basis];
    }

  {
    DynamicSparsityPattern dsp((this->dof_handler).n_dofs());

    DoFTools::make_sparsity_pattern(
      this->dof_handler,
      dsp,
      constraints_vector[0], // sparsity pattern is the same for each basis
      /*keep_constrained_dofs =*/true); // for time stepping this is essential
                                        // to be true
    sparsity_pattern.copy_from(dsp);
  }

  //  system_matrix.reinit(sparsity_pattern);
  //  system_matrix_with_constraints.reinit(sparsity_pattern);

  mass_matrix.reinit(sparsity_pattern);
  advection_matrix.reinit(sparsity_pattern);
  advection_matrix_old.reinit(sparsity_pattern);
  diffusion_matrix.reinit(sparsity_pattern);
  diffusion_matrix_old.reinit(sparsity_pattern);

  //  system_rhs.reinit((this->dof_handler).n_dofs());
  //  tmp.reinit((this->dof_handler).n_dofs());

  global_rhs.reinit((this->dof_handler).n_dofs());
  global_rhs_old.reinit((this->dof_handler).n_dofs());

  reconstructed_solution.reinit((this->dof_handler).n_dofs());
}


template <int dim>
void H1Conformal<dim>::assemble_system(
  const double current_time,
  const double time_step)
{
  Assert(is_initialized, ExcNotInitialized());

  if (rebuild_system_matrix)
    {
      const QGauss<dim> quadrature_formula((this->fe).degree + 1);

      FEValues<dim> fe_values(this->fe,
                              quadrature_formula,
                              update_values | update_gradients |
                                update_quadrature_points | update_JxW_values);

      const unsigned int dofs_per_cell = (this->fe).dofs_per_cell;
      const unsigned int n_q_points    = quadrature_formula.size();

      FullMatrix<double> cell_matrix_mass(dofs_per_cell, dofs_per_cell);
      FullMatrix<double> cell_matrix_advection(dofs_per_cell, dofs_per_cell);
      FullMatrix<double> cell_matrix_advection_old(dofs_per_cell,
                                                   dofs_per_cell);
      FullMatrix<double> cell_matrix_diffusion(dofs_per_cell, dofs_per_cell);
      FullMatrix<double> cell_matrix_diffusion_old(dofs_per_cell,
                                                   dofs_per_cell);
      Vector<double>     cell_rhs(dofs_per_cell);
      Vector<double>     cell_rhs_old(dofs_per_cell);

      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

      std::vector<Tensor<2, dim>> matrix_coeff_values_old(n_q_points);
      std::vector<Tensor<2, dim>> matrix_coeff_values(n_q_points);

      std::vector<Tensor<1, dim>> advection_field_values_old(n_q_points);
      std::vector<Tensor<1, dim>> advection_field_values(n_q_points);

      std::vector<double> rhs_values_old(n_q_points);
      std::vector<double> rhs_values(n_q_points);

      for (const auto &cell : (this->dof_handler).active_cell_iterators())
        {
          cell_matrix_mass          = 0;
          cell_matrix_advection     = 0;
          cell_matrix_advection_old = 0;
          cell_matrix_diffusion     = 0;
          cell_matrix_diffusion_old = 0;
          cell_rhs                  = 0;
          cell_rhs_old              = 0;

          fe_values.reinit(cell);
          cell->get_dof_indices(local_dof_indices);

          const std::vector<Point<dim>> quad_points(
            fe_values.get_quadrature_points());

          /*
           * Values at current time.
           */
          this->matrix_coeff().set_time(current_time);
          this->advection_field().set_time(current_time);
          this->right_hand_side().set_time(current_time);

          this->advection_field().value_list(quad_points,
                                             advection_field_values);
          this->matrix_coeff().value_list(fe_values.get_quadrature_points(),
                                          matrix_coeff_values);
          this->right_hand_side().value_list(fe_values.get_quadrature_points(),
                                             rhs_values);

          /*
           * Values at previous time.
           */
          this->matrix_coeff().set_time(current_time - time_step);
          this->advection_field().set_time(current_time - time_step);
          this->right_hand_side().set_time(current_time - time_step);
          this->advection_field().value_list(fe_values.get_quadrature_points(),
                                             advection_field_values_old);
          this->matrix_coeff().value_list(fe_values.get_quadrature_points(),
                                          matrix_coeff_values_old);
          this->right_hand_side().value_list(fe_values.get_quadrature_points(),
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
                      // on LHS
                      cell_matrix_mass(i, j) +=
                        fe_values.shape_value(i, q_index) *
                        fe_values.shape_value(j, q_index) *
                        fe_values.JxW(q_index);
                      // on LHS
                      cell_matrix_advection(i, j) +=
                        fe_values.shape_value(i, q_index) *
                        advection_field_values[q_index] *
                        fe_values.shape_grad(j, q_index) *
                        fe_values.JxW(q_index);
                      // on LHS
                      cell_matrix_advection_old(i, j) +=
                        fe_values.shape_value(i, q_index) *
                        advection_field_values_old[q_index] *
                        fe_values.shape_grad(j, q_index) *
                        fe_values.JxW(q_index);
                      // on RHS (note the sign)
                      cell_matrix_diffusion(i, j) -=
                        fe_values.shape_grad(i, q_index) *
                        matrix_coeff_values[q_index] *
                        fe_values.shape_grad(j, q_index) *
                        fe_values.JxW(q_index);
                      // on RHS (note the sign)
                      cell_matrix_diffusion_old(i, j) -=
                        fe_values.shape_grad(i, q_index) *
                        matrix_coeff_values_old[q_index] *
                        fe_values.shape_grad(j, q_index) *
                        fe_values.JxW(q_index);
                    } // ++j
                  // on RHS
                  cell_rhs(i) += fe_values.shape_value(i, q_index) *
                                 rhs_values[q_index] * fe_values.JxW(q_index);
                  // on RHS
                  cell_rhs_old(i) += fe_values.shape_value(i, q_index) *
                                     rhs_values_old[q_index] *
                                     fe_values.JxW(q_index);
                } // ++i
            }     // ++q_index
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
                  mass_matrix.add(local_dof_indices[i],
                                  local_dof_indices[j],
                                  cell_matrix_mass(i, j));
                  advection_matrix.add(local_dof_indices[i],
                                       local_dof_indices[j],
                                       cell_matrix_advection(i, j));
                  advection_matrix_old.add(local_dof_indices[i],
                                           local_dof_indices[j],
                                           cell_matrix_advection_old(i, j));
                  diffusion_matrix.add(local_dof_indices[i],
                                       local_dof_indices[j],
                                       cell_matrix_diffusion(i, j));
                  diffusion_matrix_old.add(local_dof_indices[i],
                                           local_dof_indices[j],
                                           cell_matrix_diffusion_old(i, j));
                }
              global_rhs(local_dof_indices[i]) += cell_rhs(i);
              global_rhs_old(local_dof_indices[i]) += cell_rhs_old(i);
            }
        } // ++cell
    }     // if rebuild_system_matrix

  rebuild_system_matrix = (this->advection_field().is_transient) ||
                          (this->matrix_coeff().is_transient) ||
                          (this->right_hand_side().is_transient);
}


template <int dim>
void H1Conformal<dim>::compute_time_derivative(const double time_step)
{
  Assert(is_initialized, ExcNotInitialized());

  for (unsigned int index_basis = 0;
       index_basis < GeometryInfo<dim>::vertices_per_cell;
       ++index_basis)
    {
      (this->solution_vector_time_derivative())[index_basis] = 0;

      (this->solution_vector_time_derivative())[index_basis] +=
        (this->solution_vector())[index_basis];
      (this->solution_vector_time_derivative())[index_basis] -=
        (this->solution_vector_old())[index_basis];

      (this->solution_vector_time_derivative())[index_basis] /= time_step;
    }
}


template <int dim>
void H1Conformal<dim>::assemble_global_element_data(FullMatrix<double> &global_element_matrix,
                               FullMatrix<double> &global_element_matrix_old,
                               Vector<double> &    global_element_rhs,
                               Vector<double> &    global_element_rhs_old,
                               const double        time_step,
                               const double        theta)
{
  Assert(is_initialized, ExcNotInitialized());

  /*
   * We assemble the global system Mu_t + Cu = Au + f. In time discrete
   * form this reads (Mu)' - Nu + Cu = Au + f.
   * With the theta-method and (Nu)_i = <phi_i', phi_j>u_j which amounts to
   * [M^{n+1} + dt*theta*(C-N-A)]u^{n+1} = [M^n + dt*(1-theta)*(A+N-C)]u^n
   * + theta*f^{n+1} + (1-theta)*f^n
   */

  Assert(is_initialized, ExcNotInitialized());

  compute_time_derivative(time_step);

  // First reset
  global_element_matrix     = 0;
  global_element_matrix_old = 0;
  global_element_rhs        = 0;
  global_element_rhs_old    = 0;

  {
    // Mass matrix
    assemble_global_element_matrix(
      mass_matrix,
      global_element_matrix,
      /* factor */ 1,
      /* use_time_derivative_test_function */ false,
      /* at_current_time_step */ true);

    // Mass matrix at previous time
    assemble_global_element_matrix(
      mass_matrix,
      global_element_matrix_old,
      /* factor */ 1,
      /* use_time_derivative_test_function */ false,
      /* at_current_time_step */ false);
  }

  if (!Timedependent_AdvectionDiffusionProblemUtilities::is_approx(theta, 0.0))
    {
      /*
       * Means we do not have a fully explicit method
       * so that we must assemble more than just mass
       * for the system matrix.
       */

      // Mass matrix derivative N
      assemble_global_element_matrix(
        mass_matrix,
        global_element_matrix,
        /* factor */ (-1) * theta * time_step,
        /* use_time_derivative_test_function */ true,
        /* at_current_time_step */ true);

      // Advection matrix C
      assemble_global_element_matrix(
        advection_matrix,
        global_element_matrix,
        /* factor */ theta * time_step,
        /* use_time_derivative_test_function */ false,
        /* at_current_time_step */ true);

      // Diffusion matrix D
      assemble_global_element_matrix(
        diffusion_matrix,
        global_element_matrix,
        /* factor */ (-1) * theta * time_step,
        /* use_time_derivative_test_function */ false,
        /* at_current_time_step */ true);

      // Forcing at current time step
      assemble_global_element_rhs(global_rhs,
                                  global_element_rhs,
                                  theta,
                                  /* at_current_time_step */ true);
    }

  if (!Timedependent_AdvectionDiffusionProblemUtilities::is_approx(theta, 1.0))
    {
      /*
       * Means we do not have a fully implicit method
       * so that we must assemble more than just mass
       * for the system matrix.
       */

      // Mass matrix derivative N at previous time step
      assemble_global_element_matrix(
        mass_matrix,
        global_element_matrix_old,
        /* factor */ (1 - theta) * time_step,
        /* use_time_derivative_test_function */ true,
        /* at_current_time_step */ false);

      // Advection matrix C at previous time step
      assemble_global_element_matrix(
        advection_matrix_old,
        global_element_matrix_old,
        /* factor */ (-1) * (1 - theta) * time_step,
        /* use_time_derivative_test_function */ false,
        /* at_current_time_step */ false);

      // Diffusion matrix A at previous time step
      assemble_global_element_matrix(
        diffusion_matrix_old,
        global_element_matrix_old,
        /* factor */ (1 - theta) * time_step,
        /* use_time_derivative_test_function */ false,
        /* at_current_time_step */ false);

      // Forcing at previous time step
      assemble_global_element_rhs(global_rhs_old,
                                  global_element_rhs_old,
                                  (1 - theta),
                                  /* at_current_time_step */ false);
    }
}


template <int dim>
void H1Conformal<dim>::
  assemble_global_element_matrix(const SparseMatrix<double> &relevant_matrix,
                                 FullMatrix<double> &        global_data_matrix,
                                 const double                factor,
                                 const bool use_time_derivative_test_function,
                                 const bool at_current_time_step)
{
  Assert(is_initialized, ExcNotInitialized());

  // Get lengths of tmp vectors for assembly
  const unsigned int dofs_per_cell = (this->fe).n_dofs_per_cell();

  Vector<double> tmp((this->dof_handler).n_dofs());

  std::vector<Vector<double>> &relevant_test_vector =
    (at_current_time_step ? (use_time_derivative_test_function ?
                               this->solution_vector_time_derivative() :
                               this->solution_vector()) :
                            (use_time_derivative_test_function ?
                               this->solution_vector_time_derivative() :
                               this->solution_vector_old()));
  std::vector<Vector<double>> &relevant_trial_vector =
    (at_current_time_step ? this->solution_vector() :
                            this->solution_vector_old());

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
    }     // end for i_test
}


template <int dim>
void H1Conformal<dim>::assemble_global_element_rhs(const Vector<double> &local_forcing,
                                    Vector<double> &      global_data_vector,
                                    const double          factor,
                                    const bool            at_current_time_step)
{
  Assert(is_initialized, ExcNotInitialized());

  // Get lengths of tmp vectors for assembly
  const unsigned int dofs_per_cell = (this->fe).n_dofs_per_cell();

  std::vector<Vector<double>> &relevant_test_vector =
    (at_current_time_step ? this->solution_vector() :
                            this->solution_vector_old());

  // This assembles the local contribution to the global global matrix
  // with an algebraic trick. It uses the local system matrix stored in
  // the respective basis object.
  for (unsigned int i_test = 0; i_test < dofs_per_cell; ++i_test)
    {
      // set an alias name
      const Vector<double> &test_vec = relevant_test_vector[i_test];

      global_data_vector(i_test) += test_vec * local_forcing;
      global_data_vector(i_test) *= factor;

    } // end for i_test
}


template <int dim>
void H1Conformal<dim>::project_reconstructed_function(double current_time,
                                       double previous_time)
{
  CellId cell_id = this->local_basis_ptr->get_global_cell_id();
  //  if (this->is_first_cell)
  {
    std::cout << "-----------------------------------" << std::endl
              << "MPI rank:   "
              << Utilities::MPI::this_mpi_process(this->mpi_communicator)
              << " / "
              << Utilities::MPI::n_mpi_processes(this->mpi_communicator)
              << std::endl
              << "Basis info:   "
              << this->local_basis_ptr->get_basis_info_string() << std::endl;

    std::cout << "Local number of DoFs:   " << this->dof_handler.n_dofs()
              << std::endl
              << "CellId:   " << cell_id.to_string() << std::endl;

    /*
     * Project the global FE function of the previous time step onto the
     * traced back mesh.
     */
    VectorTools::project(this->dof_handler,
                         constraints_fake,
                         QGauss<dim>((this->fe).degree + 1),
                         local_to_global_field_function,
                         reconstructed_solution);
  }

  ////////////////////////////////////////////////////////////
  /// Print only mesh in first cell
  if (this->is_first_cell)
    {
      const std::string filename_trace_back(
        "traced_local_cell_id-" + cell_id.to_string() + "_time-" +
        Utilities::to_string(current_time, 3) + "_to-" +
        Utilities::to_string(previous_time, 3) + ".vtu");

      this->print_traced_mesh_info(reconstructed_solution, filename_trace_back);
    }
  ////////////////////////////////////////////////////////////

  std::cout << "-----------------------------------" << std::endl << std::endl;
}


template <int dim>
void
H1Conformal<dim>::basis_initial_reconstruction()
{
  Assert(is_initialized, ExcNotInitialized());

  //
  // TODO. For now initialized as standard Q1 basis.
  //
 basis_q1.initialize();


//  const Tensor<1, dim> edge_directions[2] = {_global_cell->vertex(1) -
//		  _global_cell->vertex(0),
//		  _global_cell->vertex(2) -
//		  _global_cell->vertex(0)};
//  const Tensor<2, dim> edge_tensor(
//    {{edge_directions[0][0], edge_directions[0][1]},
//     {edge_directions[1][0], edge_directions[1][1]}});
//  const bool is_right_handed_cell = (determinant(edge_tensor) > 0);

 double  k_edge = 0.01;
 for (const auto &cell : (this->dof_handler).active_cell_iterators())
        {
	 std::vector<Vector<double>> *solution_vector_old_ptr;
	 auto a_1 = &solution_vector_old_ptr[1];
	 auto   a_2 = &solution_vector_old_ptr[2];
  		system_matrix[0][0] =  (a_1^2 + k_edge);
  		system_matrix[0][1] =(a_1*a_2);
  		system_matrix[1][0] = (a_1*a_2);
  		system_matrix[1][1] =(a_2^2 + k_edge);


// auto 	rhs = ( [k_edge*basis_left + a_1*uOrigEdge ;
//  		            k_edge*basis_right + a_2*uOrigEdge]
//  		            - system_matrix[:,ind_con]*val_con )
        }
  for (unsigned int index_basis = 0;
         index_basis < GeometryInfo<dim>::vertices_per_cell;
         ++index_basis)
      {
        basis_q1.set_index(index_basis);

      }
  /*
   * An optimal basis must be found at the initial time through solving a
   * (regularized) inverse problem on the traced back mesh. Edge data must be
   * reconstructed first. One can in principle also put measurement data into
   * the inverse problem.
   */
  //  solve_basis_optimization();
}


template <int dim>
void H1Conformal<dim>::basis_reconstruction(const double current_time, // time at which basis
                                                        // needs reconstruction
                             const double time_step,
                             const double /* theta */)
{
  Assert(is_initialized, ExcNotInitialized());

  /*
   * The mesh needs to be traced back each time step (unless the advection field
   * is constant)
   */
  this->trace_back_mesh(current_time);

  /*
   * The global (fine scale) solution at the previous time step needs to be
   * reconstructed on the traced back mesh. Note that MPI communication may be
   * necessary since relevant information can be located on another MPI rank.
   * The result serves as data for the following inverse problem.
   */
  project_reconstructed_function(current_time,
                                 /* previous time */ current_time - time_step);

  /*
   * An optimal basis must be found at the previous time step through solving a
   * (regularized) inverse problem on the traced back mesh. Edge data must be
   * reconstructed first. One can in principle also put measurement data into
   * the inverse problem.
   */
  //  solve_basis_optimization();

  /*
   * The reconstructed basis must be mapped forward in time according to the PDE
   * at hand to the next time step (which is actually current_time). Note that
   * for compressible problems (or problems with a reaction term) boundary data
   * should be evolved first through a (heuristically) reduced PDE.
   */
  //  advance_reconstructed_basis();


  /*
   * This is to speed up the global assembly.
   */
  assemble_system(current_time, time_step);
}


template <int dim>
const std::string H1Conformal<dim>::get_info_string()
{
  return "_reconstruction-h1conformal";
}


}

#endif /* INCLUDE_H1_CONFORMAL_HPP_ */
