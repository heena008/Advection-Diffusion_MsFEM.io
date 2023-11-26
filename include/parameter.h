/*
 * parameter.h
 *
 *  Created on: Jan 13, 2021
 *      Author: heena
 */

#ifndef PROJECT_INCLUDE_PARAMETER_H_
#define PROJECT_INCLUDE_PARAMETER_H_

#include <deal.II/base/parameter_handler.h>

#include <fstream>
#include <iostream>
#include <memory>
/*!
 * @class Diffusion Coefficient
 * @brief Class implements Diffusion coefficient
 */
using namespace dealii;
namespace Parameters
{

  struct Geometry
  {

    double       scale;

    static void declare_parameters(ParameterHandler &prm);
    void parse_parameters(ParameterHandler &prm);
  };

  void Geometry::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Geometry");
    {

      prm.declare_entry("Grid scale",
                        "1e-3",
                        Patterns::Double(0.0),
                        "Global grid scaling factor");

    }
    prm.leave_subsection();
  }
  void Geometry::parse_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Geometry");
    {

      scale  = prm.get_double("Grid scale");

    }
    prm.leave_subsection();
  }

  struct ReferenceValues
   {

     static void
     declare_parameters(ParameterHandler &prm);

     void
     parse_parameters(ParameterHandler &prm);

     /*!
      * Reference time is in second
      */
     double time; /* s */

     /*!
      * Reference velocity.
      */
     double velocity; /* m/s */

     /*!
      * Reference length.
      */
    double length; /* m */
   };


    void ReferenceValues::declare_parameters(ParameterHandler &prm)
    {

        prm.enter_subsection("ReferenceValues");
        {
          prm.declare_entry("velocity",
                            "10",
                            Patterns::Double(0),
                            "Reference velocity");

          prm.declare_entry("length",
                            "1e+4",
                            Patterns::Double(0),
                            "Reference length.");

        }
        prm.leave_subsection();
      }



    void
   ReferenceValues::parse_parameters(ParameterHandler &prm)
    {

        prm.enter_subsection("ReferenceValues");
        {
          velocity           = prm.get_double("velocity");           /* m/s */
          length             = prm.get_double("length");             /* m */

        }
        prm.leave_subsection();
        time = length / velocity; /* s */
      }

//  struct Materials
//  {
//    double nu;
//    double mu;
//    static void declare_parameters(ParameterHandler &prm);
//    void parse_parameters(ParameterHandler &prm);
//  };
//  void Materials::declare_parameters(ParameterHandler &prm)
//  {
//    prm.enter_subsection("Material properties");
//    {
//      prm.declare_entry("Poisson's ratio",
//                        "0.4999",
//                        Patterns::Double(-1.0, 0.5),
//                        "Poisson's ratio");
//      prm.declare_entry("Shear modulus",
//                        "80.194e6",
//                        Patterns::Double(),
//                        "Shear modulus");
//    }
//    prm.leave_subsection();
//  }
//  void Materials::parse_parameters(ParameterHandler &prm)
//  {
//    prm.enter_subsection("Material properties");
//    {
//      nu = prm.get_double("Poisson's ratio");
//      mu = prm.get_double("Shear modulus");
//    }
//    prm.leave_subsection();
//  }
//  struct LinearSolver
//  {
//    std::string type_lin;
//    double      tol_lin;
//    double      max_iterations_lin;
//    bool        use_static_condensation;
//    std::string preconditioner_type;
//    double      preconditioner_relaxation;
//    static void declare_parameters(ParameterHandler &prm);
//    void parse_parameters(ParameterHandler &prm);
//  };
//  void LinearSolver::declare_parameters(ParameterHandler &prm)
//  {
//    prm.enter_subsection("Linear solver");
//    {
//      prm.declare_entry("Solver type",
//                        "CG",
//                        Patterns::Selection("CG|Direct"),
//                        "Type of solver used to solve the linear system");
//      prm.declare_entry("Residual",
//                        "1e-6",
//                        Patterns::Double(0.0),
//                        "Linear solver residual (scaled by residual norm)");
//      prm.declare_entry(
//        "Max iteration multiplier",
//        "1",
//        Patterns::Double(0.0),
//        "Linear solver iterations (multiples of the system matrix size)");
//      prm.declare_entry("Use static condensation",
//                        "true",
//                        Patterns::Bool(),
//                        "Solve the full block system or a reduced problem");
//      prm.declare_entry("Preconditioner type",
//                        "ssor",
//                        Patterns::Selection("jacobi|ssor"),
//                        "Type of preconditioner");
//      prm.declare_entry("Preconditioner relaxation",
//                        "0.65",
//                        Patterns::Double(0.0),
//                        "Preconditioner relaxation value");
//    }
//    prm.leave_subsection();
//  }
//  void LinearSolver::parse_parameters(ParameterHandler &prm)
//  {
//    prm.enter_subsection("Linear solver");
//    {
//      type_lin                  = prm.get("Solver type");
//      tol_lin                   = prm.get_double("Residual");
//      max_iterations_lin        = prm.get_double("Max iteration multiplier");
//      use_static_condensation   = prm.get_bool("Use static condensation");
//      preconditioner_type       = prm.get("Preconditioner type");
//      preconditioner_relaxation = prm.get_double("Preconditioner relaxation");
//    }
//    prm.leave_subsection();
//  }
//  struct NonlinearSolver
//  {
//    unsigned int max_iterations_NR;
//    double       tol_f;
//    double       tol_u;
//    static void declare_parameters(ParameterHandler &prm);
//    void parse_parameters(ParameterHandler &prm);
//  };
//  void NonlinearSolver::declare_parameters(ParameterHandler &prm)
//  {
//    prm.enter_subsection("Nonlinear solver");
//    {
//      prm.declare_entry("Max iterations Newton-Raphson",
//                        "10",
//                        Patterns::Integer(0),
//                        "Number of Newton-Raphson iterations allowed");
//      prm.declare_entry("Tolerance force",
//                        "1.0e-9",
//                        Patterns::Double(0.0),
//                        "Force residual tolerance");
//      prm.declare_entry("Tolerance displacement",
//                        "1.0e-6",
//                        Patterns::Double(0.0),
//                        "Displacement error tolerance");
//    }
//    prm.leave_subsection();
//  }
//  void NonlinearSolver::parse_parameters(ParameterHandler &prm)
//  {
//    prm.enter_subsection("Nonlinear solver");
//    {
//      max_iterations_NR = prm.get_integer("Max iterations Newton-Raphson");
//      tol_f             = prm.get_double("Tolerance force");
//      tol_u             = prm.get_double("Tolerance displacement");
//    }
//    prm.leave_subsection();
//  }
//  struct Time
//  {
//    double delta_t;
//    double end_time;
//    static void declare_parameters(ParameterHandler &prm);
//    void parse_parameters(ParameterHandler &prm);
//  };
//  void Time::declare_parameters(ParameterHandler &prm)
//  {
//    prm.enter_subsection("Time");
//    {
//      prm.declare_entry("End time", "1", Patterns::Double(), "End time");
//      prm.declare_entry("Time step size",
//                        "0.1",
//                        Patterns::Double(),
//                        "Time step size");
//    }
//    prm.leave_subsection();
//  }
//  void Time::parse_parameters(ParameterHandler &prm)
//  {
//    prm.enter_subsection("Time");
//    {
//      end_time = prm.get_double("End time");
//      delta_t  = prm.get_double("Time step size");
//    }
//    prm.leave_subsection();
//  }
  struct AllParameters :  public Geometry,
                          public ReferenceValues

  {
    AllParameters(const std::string &input_file);
    static void declare_parameters(ParameterHandler &prm);
    void parse_parameters(ParameterHandler &prm);
  };
  AllParameters::AllParameters(const std::string &input_file)
  {
    ParameterHandler prm;
    declare_parameters(prm);
    prm.parse_input(input_file);
    parse_parameters(prm);
  }
  void AllParameters::declare_parameters(ParameterHandler &prm)
  {

    Geometry::declare_parameters(prm);
    ReferenceValues::declare_parameters(prm);
  }
  void AllParameters::parse_parameters(ParameterHandler &prm)
  {

    Geometry::parse_parameters(prm);
    ReferenceValues::declare_parameters(prm);

  }
} // namespace Parameters





#endif /* PROJECT_INCLUDE_PARAMETER_H_ */
