/*
 * advection_diffusion_parameter.h
 *
 *  Created on: Jan 14, 2021
 *      Author: heena
 */

#ifndef INCLUDE_ADVECTION_DIFFUSION_PARAMETER_H_
#define INCLUDE_ADVECTION_DIFFUSION_PARAMETER_H_

// Deal.ii
#include <deal.II/base/parameter_handler.h>


#include "referencevalue.h"
#include <fstream>
#include <iostream>
#include <memory>

using namespace dealii;
namespace Coefficient
{

  struct Nondimensional
  {
	  Nondimensional(const std::string &parameter_filename);

	  static void declare_parameters(ParameterHandler &prm);
	     void parse_parameters(ParameterHandler &prm);

   Coefficient::ReferenceValues reference_values;


  };

  Coefficient::ReferenceValues::ReferenceValues(
    const std::string &parameter_filename)
    : time(0)
    , velocity(10)
    , length(1e+4)
  {
    ParameterHandler prm;
    ReferenceValues::declare_parameters(prm);

    std::ifstream parameter_file(parameter_filename);
    if (!parameter_file)
      {
        parameter_file.close();
        std::ofstream parameter_out(parameter_filename);
        prm.print_parameters(parameter_out, ParameterHandler::Text);
        AssertThrow(false,
                    ExcMessage(
                      "Input parameter file <" + parameter_filename +
                      "> not found. Creating a template file of the same name."));
      }
    prm.parse_input(parameter_file,
                    /* filename = */ "generated_parameter.in",
                    /* last_line = */ "",
                    /* skip_undefined = */ true);
    parse_parameters(prm);
  }



  void
  Coefficient::ReferenceValues::declare_parameters(ParameterHandler &prm)
  {

      prm.enter_subsection("Reference quantities");
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
  Coefficient::ReferenceValues::parse_parameters(ParameterHandler &prm)
  {

      prm.enter_subsection("Reference quantities");
      {
        velocity           = prm.get_double("velocity");           /* m/s */
        length             = prm.get_double("length");             /* m */
        time = length / velocity; /* s */
      }
      prm.leave_subsection();
    }






}




#endif /* INCLUDE_ADVECTION_DIFFUSION_PARAMETER_H_ */
