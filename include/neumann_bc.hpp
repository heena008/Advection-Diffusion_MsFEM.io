/*
 * neumann_bc.hpp
 *
 *  Created on: Oct 7, 2019
 *      Author: heena
 */

#ifndef INCLUDE_NEUMANN_BC_HPP_
#define INCLUDE_NEUMANN_BC_HPP_

// Deal.ii
#include <deal.II/base/function.h>

// STL
#include <cmath>
#include <fstream>

// My Headers
#include "coefficients.h"

namespace Coefficients {
using namespace dealii;

/*!
 * @class NeumannBC
 * @brief Class implements scalar Neumann conditions.
 */
template <int dim> class NeumannBC : public Function<dim>
{
public:
  NeumannBC() : Function<dim>() {}

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;
  virtual void value_list(const std::vector<Point<dim>> &points,
                          std::vector<double> &values,
                          const unsigned int component = 0) const override;
};

template <int dim>
double NeumannBC<dim>::value(const Point<dim> &p,
                             const unsigned int /*component*/) const
{
//  double return_value = 0;
//
//  return return_value;


	double A = 1;
		     double B = 1;


	       const double x_minus_mui = p[1] -60;


	       double sum= 0;//A*std::exp(-B * (x_minus_mui)* (x_minus_mui));




	return sum;
}

template <int dim>
void NeumannBC<dim>::value_list(const std::vector<Point<dim>> &points,
                                std::vector<double> &values,
                                const unsigned int /*component = 0*/) const
{
  Assert(points.size() == values.size(),
         ExcDimensionMismatch(points.size(), values.size()));

  for (unsigned int p = 0; p < points.size(); ++p) {
    values[p] = value(points[p]);
  } // end ++p
}

} // end namespace Coefficients

#endif /* INCLUDE_NEUMANN_BC_HPP_ */
