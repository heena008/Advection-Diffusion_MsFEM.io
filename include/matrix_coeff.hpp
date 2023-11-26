/*
 * matrix_coeff.hpp
 *
 *  Created on: Oct 7, 2019
 *      Author: heena
 */

#ifndef INCLUDE_MATRIX_COEFF_HPP_
#define INCLUDE_MATRIX_COEFF_HPP_

// Deal.ii
#include <deal.II/base/tensor_function.h>

// STL
#include <cmath>
#include <fstream>

// My Headers
#include "coefficients.h"

namespace Coefficients {
using namespace dealii;

/*!
 * @class MatrixCoeff
 * @brief Diffusion coefficient.
 *
 * Class implements a matrix valued diffusion coefficient.
 * This coefficient must be positive definite.
 */

template <int dim> class MatrixCoeff : public TensorFunction<2, dim>
{
public:
  MatrixCoeff() : TensorFunction<2, dim>() {}

  virtual Tensor<2, dim> value(const Point<dim> &point) const override;
  virtual void value_list(const std::vector<Point<dim>> &points,
                          std::vector<Tensor<2, dim>> &values) const override;


  const bool is_transient = false;


};

template <int dim>
Tensor<2, dim> MatrixCoeff<dim>::value(const Point<dim> &p) const
{
  Tensor<2, dim> value;
  value.clear();

  const double t = this->get_time();

  for (unsigned int d = 0; d < dim; ++d)
  {
//	  value[d][d] = 0.01; /* Must be positive definite. For non-periodic case*/
     value[d][d] =0.1*(1- 0.9999 * sin(60 * pi *  p(d)));

 /* Must be positive
    //  definite.For periodic case */
  }

//  Point<dim> new_position = p;
//
//  double dtheta = -45;
//
//
//     	         new_position[0] = std::cos(dtheta) * p[0] - std::sin(dtheta) * p[1];
//     	         new_position[1] = std::sin(dtheta) * p[0] + std::cos(dtheta) * p[1];

//  for (unsigned int d = 0; d < dim; ++d)
//   {
////  if (p[0]>=-25 && p[0]<=25 && p[1]<=30)
//	  if (new_position[0] >=-13.13   && new_position[0] <=38   && new_position[1]  >=21.27
//	    		 && new_position[1] <=51  )
//	  {
// 		                 value[d][d] =3000;//k*exp(-beta*(1-(p[1]/h)));
// 		               }
//
// 	                else {
// 	                	value[d][d] =4*(p[1]);
// 	               }

//}

  return value;
}

template <int dim>
void MatrixCoeff<dim>::value_list(const std::vector<Point<dim>> &points,
                                  std::vector<Tensor<2, dim>> &values) const
{
  Assert(points.size() == values.size(),
         ExcDimensionMismatch(points.size(), values.size()));

  for (unsigned int p = 0; p < points.size(); ++p)
  {
    values[p].clear();

    values[p] = value(points[p]);
  }
}

} // end namespace Coefficients
#endif /* INCLUDE_MATRIX_COEFF_HPP_ */
