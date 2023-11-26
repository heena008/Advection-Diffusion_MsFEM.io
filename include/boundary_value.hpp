/*
 * boundary_value.hpp
 *
 *  Created on: Oct 8, 2020
 *      Author: heena
 */

#ifndef INCLUDE_BOUNDARY_VALUE_HPP_
#define INCLUDE_BOUNDARY_VALUE_HPP_

// Deal.ii
#include <deal.II/base/function.h>

// STL
#include <cmath>
#include <fstream>

// My Headers
#include "coefficients.h"

namespace Coefficients {
using namespace dealii;

template <int dim> class BoundaryValues : public Function<dim>
{
public:
	BoundaryValues() : Function<dim>() {}

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;
  virtual void value_list(const std::vector<Point<dim>> &points,
                          std::vector<double> &values,
                          const unsigned int component = 0) const override;
};

template <int dim>
double BoundaryValues<dim>::value(const Point<dim> & p,
                                  const unsigned int component) const
{
  Assert(component == 0, ExcIndexRange(component, 0, 1));
  (void)component;
  if (std::fabs(p[0] - 1) < 1e-8 ||
      (std::fabs(p[1] + 1) < 1e-8 && p[0] >= 0.5))
    {
      return 1.0;
    }
  else
    {
      return 0.0;
    }
}

template <int dim>
void BoundaryValues<dim>::value_list(const std::vector<Point<dim>> &points,
                                     std::vector<double> &          values,
                                     const unsigned int component) const
{
  Assert(values.size() == points.size(),
         ExcDimensionMismatch(values.size(), points.size()));
  for (unsigned int i = 0; i < points.size(); ++i)
    values[i] = BoundaryValues<dim>::value(points[i], component);
}


} // end namespace Coefficients



#endif /* INCLUDE_BOUNDARY_VALUE_HPP_ */
