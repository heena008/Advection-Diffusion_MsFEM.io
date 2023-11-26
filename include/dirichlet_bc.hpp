/*
 * dirichlet_bc.hpp
 *
 *  Created on: Oct 7, 2019
 *      Author: heena
 */

#ifndef INCLUDE_DIRICHLET_BC_HPP_
#define INCLUDE_DIRICHLET_BC_HPP_

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
 * @class DirichletBC
 * @brief Class implements Dirichlet conditions.
 */
template <int dim> class DirichletBC : public Function<dim>
{
public:
  DirichletBC() : Function<dim>() {}

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;
  virtual void value_list(const std::vector<Point<dim>> &points,
                          std::vector<double> &values,
                          const unsigned int component = 0) const override;

protected :
  const unsigned int n_mu = 2;
  	  const Point<dim> mu[2]={Point<2>(1. / 3, 0.5), Point<2>(2. / 3, 0.5)};
};

template <int dim>
double DirichletBC<dim>::value(const Point<dim> &p,
                               const unsigned int /*component*/) const
{
  double return_value = 0;

  double sum = 0;
   	    Tensor<2, 2> M;
   	    for (unsigned int d = 0; d < dim; ++d) {
   	      M[d][d] = 0.03;
   	    }

   	    double det_M = determinant(M);
   	    for (unsigned int i = 0; i < this->n_mu; ++i)
   	    {
   	      const Tensor<1, dim> x_minus_mui = p - this->mu[i];
   	      sum += std::exp(-0.5 * (x_minus_mui)*invert(M) * (x_minus_mui));
   	    }
   	    return_value = sum / (2 * std::sqrt(4 *pi * pi * det_M));


//  for (unsigned int d = 0; d < dim; ++d)
//    return_value += (p(d) - 0.5) * (p(d) - 0.5);

  // new test cases

//  {
//    (void)component;
//    Assert(component == 0, ExcIndexRange(component, 0, 1));
//    int i=1;
//    if (i==1)
//
//      return_value= 5*sin(5*pi*(p[0]))*sin(5*pi*p[1]);
//
//		  else if(i==2)
//		  {
//			  return_value= 10*sin(2*pi*(p[0]))*sin(2*pi*p[1]);
//		  }
//    else
//    	 return_value= 0;


//
//  Point<dim> new_position = p;
//
//
//      double dtheta = -45;
//
//
//      	         new_position[0] = std::cos(dtheta) * p[0] - std::sin(dtheta) * p[1];
//      	         new_position[1] = std::sin(dtheta) * p[0] + std::cos(dtheta) * p[1];
//
////      	       if (p[0]>=-25 && p[0]<=25 && p[1]<=30)
//      	       if (new_position[0] >=-13.13   && new_position[0] <=38  && new_position[1]  >=21.27
//      	    		 && new_position[1] <=51    )
//      	         {
//		            	   return_value =0;
//			               }
//
//	 			else
//	 				return_value =1;

  return return_value;

}

template <int dim>
void DirichletBC<dim>::value_list(const std::vector<Point<dim>> &points,
                                  std::vector<double> &values,
                                  const unsigned int /*component = 0*/) const
{
  Assert(points.size() == values.size(),
         ExcDimensionMismatch(points.size(), values.size()));

  for (unsigned int p = 0; p < points.size(); ++p)
  {
    values[p] = value(points[p]);
  } // end ++p
}

} // end namespace Coefficients

#endif /* INCLUDE_DIRICHLET_BC_HPP_ */
