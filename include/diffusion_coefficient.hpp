/*
 * matrix_coeff.hpp
 *
 *  Created on: Oct 7, 2019
 *      Author: heena
 */

#ifndef DIFFUSION_PROBLEM_INCLUDE_DIFFUSION_COEFFICIENT_HPP_
#define DIFFUSION_PROBLEM_INCLUDE_DIFFUSION_COEFFICIENT_HPP_

// Deal.ii
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/tensor_function.h>

#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>

// STL
#include <cmath>
#include <fstream>

// My Headers
#include "coefficients.h"

namespace Coefficients
{
  using namespace dealii;
  template <int dim>
  class DiffusionCoefficient : public Function<dim>
  {
  public:
  	DiffusionCoefficient () : Function<dim>() {}

  	virtual double value(const Point<dim> &p,
  							const unsigned int component = 0) const override;
  		virtual void value_list(const std::vector<Point<dim>> &points,
  									std::vector<double>  &values,
  									const unsigned int component = 0) const override;
  };


  template <int dim>
  double
  DiffusionCoefficient<dim>::value(const Point<dim> &p,
  		const unsigned int /*component*/) const
  {
  //	Tensor<2, dim> value;
  	double value;
  	//value.clear();
  //	const double t = this->get_time();
  //
  	     double h = 30;
  	     double beta = 3;
  	     double u_star =0.35;
  	     double K_h = 10;
  	     double k = 0.4;
  	     double d_plane = 0;

  //
  //	    	 	   			value[0][0]=0.01-0.9999*sin(60*PI_D*p[0]);//Test case 1,2,3
  //	    	 	   			value[0][1]=0.01;
  //	    	 	   			value[1][0]=0.01;
  //	    	 	   			value[1][1]=0.01-0.9999*sin(60*PI_D*p[1]);

  ////
  //	             for (unsigned int d = 0; d < dim; ++d)
  //	             {
  //	            	 value[d][d]=1+t/2*(sin(2*PI_D*p[0])*sin(2*PI_D*p[1]));//New Test cases 1

  //	            	 value[d][d]=1+0.5*(sin(2*PI_D*p[0])*sin(2*PI_D*p[1]))+(1/5)*sin(2*PI_D*p[0]);//New Test cases 2



      return value;
  }

  template <int dim>
  void
  DiffusionCoefficient<dim>::value_list(const std::vector<Point<dim>> &points,
  		std::vector<double>  &values,
  		const unsigned int /*component*/) const
  {
  	Assert (points.size() == values.size(),
  			ExcDimensionMismatch (points.size(), values.size()) );

  	for ( unsigned int p=0; p<points.size(); ++p)
  	{


  		values[p] = value(points[p]);

  	}
  }


} // end namespace Coefficients
#endif /* DIFFUSION_PROBLEM_INCLUDE_DIFFUSION_COEFFICIENT_HPP_ */
