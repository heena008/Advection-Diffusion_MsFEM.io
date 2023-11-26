/*
 * runge_kutta.hpp
 *
 *  Created on: Nov 26, 2021
 *      Author: heena
 */

#ifndef PROJECT_INCLUDE_RUNGE_KUTTA_HPP_
#define PROJECT_INCLUDE_RUNGE_KUTTA_HPP_

// Deal.ii
#include <deal.II/base/tensor_function.h>

// STL
#include <cmath>
#include <fstream>

// My Headers
#include "coefficients.h"

namespace Coefficients
{
using namespace dealii;

template <int dim> class RK4 : public Tensor<1, dim> {
public:
  RK4() : Tensor<1, dim>() {}
  virtual Tensor<1, dim> value(const Point<dim> &p, double time,
                               double time_step) const;

private:
  AdvectionField<dim> advection_field;
  double T_max;
};

template <int dim>
Tensor<1, dim> RK4<dim>::value(const Point<dim> &p, double time,
                               double time_step) const {

  Tensor<1, dim> yn;

  /* Here the equation is dy/dt =c; where c is velocity y is the vertex of mesh
   * and t is the time
   *
   */

  auto k1 = time_step * advection_field.value(p),
       k2 = time_step * (advection_field.value(p) + k1 / 2),
       k3 = time_step * (advection_field.value(p) + k2 / 2),
       k4 = time_step * (advection_field.value(p) + k3);
  return yn = (1 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
}

} // end namespace Coefficients

#endif /* PROJECT_INCLUDE_RUNGE_KUTTA_HPP_ */
