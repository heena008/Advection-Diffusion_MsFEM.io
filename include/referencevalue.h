/*
 * referencevalue.h
 *
 *  Created on: Jan 14, 2021
 *      Author: heena
 */

#ifndef INCLUDE_REFERENCEVALUE_H_
#define INCLUDE_REFERENCEVALUE_H_

// Deal.ii
#include <deal.II/base/parameter_handler.h>

#include <fstream>
#include <iostream>
#include <memory>

using namespace dealii;
namespace Coefficient
{
  /*!
   * @struct ReferenceValues
   *
   * Struct contains reference quantities for non-dimensionalization
   */
  struct ReferenceValues
  {
    ReferenceValues(const std::string &input_file);

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


  }; // struct ReferenceValues

  Coefficient::ReferenceValues::ReferenceValues(
    const std::string &input_file)
    : time(0)
    , velocity(10)
    , length(1e+4)
  {
	  ParameterHandler prm;
	      declare_parameters(prm);
	      prm.parse_input(input_file);
	      parse_parameters(prm);
  }



  void
  Coefficient::ReferenceValues::declare_parameters(ParameterHandler &prm)
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
  Coefficient::ReferenceValues::parse_parameters(ParameterHandler &prm)
  {

      prm.enter_subsection("ReferenceValues");
      {
        velocity           = prm.get_double("velocity");           /* m/s */
        length             = prm.get_double("length");             /* m */

      }
      prm.leave_subsection();


    time = length / velocity; /* s */
  }



} // namespace



#endif /* INCLUDE_REFERENCEVALUE_H_ */
