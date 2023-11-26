/*
 * DiffusionProblem.cc
 *
 *  Created on: Oct 7, 2019
 *      Author: heena
 */

// My Headers
#include "advectiondiffusion_multiscale.hpp"
#include "advectiondiffusion_problem.hpp"
//#include "intlog.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include "intlog.h"


int main(int argc, char *argv[])

{

	 dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
	    argc, argv, dealii::numbers::invalid_unsigned_int);

	  dealii::deallog.depth_console(0);

	  bool         is_periodic = true;
	  unsigned int n_refine    = 4;
	  const int    dim         =2;

	          	  Timedependent_AdvectionDiffusionProblem::AdvectionDiffusionProblem<dim>
	          diffusion_problem_2d_coarse(n_refine, is_periodic);
	                diffusion_problem_2d_coarse.run();
//
//
	  Timedependent_AdvectionDiffusionProblem::AdvectionDiffusionProblem<dim>
	  						advectiondiffusion_problem(6, is_periodic);
	  			advectiondiffusion_problem.run ();


	  	   using ReconstructionType =
	  	   Timedependent_AdvectionDiffusionProblem::BasicReconstructor<dim>;

	  	  using BasisType =
	  	      Timedependent_AdvectionDiffusionProblem::SemiLagrangeBasis<dim, ReconstructionType>;

	  	 Timedependent_AdvectionDiffusionProblem::AdvectionDiffusionProblemMultiscale<dim, BasisType>diffusion_ms_problem_2d(n_refine, is_periodic);
	  	    diffusion_ms_problem_2d.run();









  return 0;
}
