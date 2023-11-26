/*
 * basis_frame_work.hpp
 *
 *  Created on: Feb 1, 2022
 *      Author: heena
 */

#ifndef INCLUDE_BASIS_FRAME_WORK_HPP_
#define INCLUDE_BASIS_FRAME_WORK_HPP_



namespace Timedependent_AdvectionDiffusionProblem {

      enum ReconstructionType
      {
        BasicReconstruction,
        H1ConformalReconstruction,
        L2ConformalReconstruction,
      };


      template <int dim>
      class  BasicReconstructor : public AdvectionDiffusionBasis_Reconstruction<dim>
      {
      public:


        virtual void
        assemble_matrix(ScratchData<dim> &   scratch_data,
                        CopyData &copy_data) = 0;


        virtual void
        assemble_rhs(TracerScratchData<dim> &   scratch_data,
                     StabilizedMethodsCopyData &copy_data) = 0;
      };


      template <int dim>
      class TracerAssemblerCore : public TracerAssemblerBase<dim>
      {
      public:
        TracerAssemblerCore(std::shared_ptr<SimulationControl> simulation_control,
                            Parameters::PhysicalProperties     physical_properties)
          : simulation_control(simulation_control)
          , physical_properties(physical_properties)
        {}

        /**
         * @brief assemble_matrix Assembles the matrix
         * @param scratch_data (see base class)
         * @param copy_data (see base class)
         */
        virtual void
        assemble_matrix(TracerScratchData<dim> &   scratch_data,
                        StabilizedMethodsCopyData &copy_data) override;


        /**
         * @brief assemble_rhs Assembles the rhs
         * @param scratch_data (see base class)
         * @param copy_data (see base class)
         */
        virtual void
        assemble_rhs(TracerScratchData<dim> &   scratch_data,
                     StabilizedMethodsCopyData &copy_data) override;

        const bool DCDD = true;

        std::shared_ptr<SimulationControl> simulation_control;
        Parameters::PhysicalProperties     physical_properties;
      };


      template <int dim>
      class TracerAssemblerBDF : public TracerAssemblerBase<dim>
      {
      public:
        TracerAssemblerBDF(std::shared_ptr<SimulationControl> simulation_control)
          : simulation_control(simulation_control)
        {}



        virtual void
        assemble_matrix(TracerScratchData<dim> &   scratch_data,
                        StabilizedMethodsCopyData &copy_data) override;


        virtual void
        assemble_rhs(TracerScratchData<dim> &   scratch_data,
                     StabilizedMethodsCopyData &copy_data) override;

        std::shared_ptr<SimulationControl> simulation_control;
      };

}

#endif /* INCLUDE_BASIS_FRAME_WORK_HPP_ */
