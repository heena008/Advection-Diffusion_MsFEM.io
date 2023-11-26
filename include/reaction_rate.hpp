/*
 * reaction_rate.hpp
 *
 *  Created on: Jul 22, 2020
 *      Author: heena
 */

#ifndef INCLUDE_REACTION_RATE_HPP_
#define INCLUDE_REACTION_RATE_HPP_

#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>

// std library
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <vector>

// My Headers
#include "coefficients.h"

namespace Coefficients {
using namespace dealii;

template <int dim> class ReactionRate : public Function<dim>
{
public:
  /*!
   * Constructor.
   */
  ReactionRate() : Function<dim>() {}
  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;
  virtual void value_list(const std::vector<Point<dim>> &points,
                          std::vector<double> &values,
                          const unsigned int component = 0) const override;
};

template <int dim>
double ReactionRate<dim>::value(const Point<dim> &p,
                                const unsigned int /*component*/) const
{
  double return_value = 0.0;

  return return_value;
}

template <int dim>
void ReactionRate<dim>::value_list(const std::vector<Point<dim>> &points, std::vector<double> &values,
								   const unsigned int /* component = 0 */) const
	{

  Assert(points.size() == values.size(),
         ExcDimensionMismatch(points.size(), values.size()));
  {
    for (unsigned int p = 0; p < points.size(); ++p)
      values[p] = 0.0;
  }
}

} // end namespace Coefficients

#endif /* INCLUDE_REACTION_RATE_HPP_ */
