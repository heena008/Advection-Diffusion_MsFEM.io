/*
 * advection_field.hpp
 *
 *  Created on: Nov 4, 2019
 *      Author: heena
 */

#ifndef INCLUDE_ADVECTION_FIELD_HPP_
#define INCLUDE_ADVECTION_FIELD_HPP_

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

/*!
 * @class AdvectionField
 * @brief Class implements Advection Field.
 */

template <int dim> class AdvectionField : public TensorFunction<1, dim>
{
public:
  AdvectionField() : TensorFunction<1, dim>() {}

  virtual Tensor<1, dim> value(const Point<dim> &point) const override;
  virtual void value_list(const std::vector<Point<dim>> &points,
						  std::vector<Tensor<1, dim>> &values) const override;


  const bool is_transient = false;



};

template <int dim>
Tensor<1, dim> AdvectionField<dim>::value(const Point<dim> &p) const
{
  const double t = this->get_time();

  Tensor<1, dim> value;
  value.clear();

//  for (unsigned int d = 0; d < dim; ++d)
//
//  {
//
//    value[d] =1;//5* std::cos( 10*numbers::PI *t);
//  }

  int i = 1;

  		if (i==1)//Test case 4.4
  		{
  			value[0] =2*pi* sin(2*pi*(p[0]))*cos(2*pi*p[1]);
  			value[1] = -2*pi*cos(2*pi*(p[0]))*sin(2*pi*p[1]);

//  						value[0] = 1;
//  						value[1] =1;
  		}
  		else if (i==2)//Test case 4.6
  		{

  			  value[0] = cos(2*pi*t)*2*pi*sin(2*pi*(p[0]-t))*cos(2*pi*p[1])-sin(2*pi*t)*cos(2*pi*(p[0]-t))*sin(2*pi*(p[1])*sin(2*pi*p[0]));
  			  value[1] = -sin(2*pi*t)*2*pi*sin(2*pi*(p[0]-t))*cos(2*pi*p[1])-cos(2*pi*t)*cos(2*pi*(p[0]-t))*sin(2*pi*(p[1])*sin(2*pi*p[0]));

  		}
  		else if (i==3)// Test case 4.7
  		{
  			  value[0] = (2*pi/5)*sin((2*pi*(p[0]- t))*(2*pi*(p[0]- t)))*cos((2*pi*(p[1]-0.5))*(2*pi*(p[1]-0.5)));
  			  value[1] =(2*pi/5)*sin(2*pi*(p[0]- t))*cos(2*pi*(p[1]-t))*cos((pi*(p[1]-0.5))*(pi*(p[1]-0.5)));
  		}

  			else
  				 std::cout
  				        << "Select the correct case" << std::endl;

   return value;

//  double alpha = 1;
//  		     double u_h = 4;
//  		     double u_star = 0.35;
//  		     double k = 0.4;
//  		   double h = 30;
//  		     double z0=0.5;
//  		   Point<dim> new_position = p;
//
//
//  		       double dtheta = -45;
//
//
//  		       	         new_position[0] = std::cos(dtheta) * p[0]- std::sin(dtheta) * p[1];
//  		       	         new_position[1] = std::sin(dtheta) * p[0] + std::cos(dtheta) * p[1];
//
//// if (p[0]>=-25  && p[0]<=25 && p[1]<=30)
//	  if (new_position[0] >=-13.13   && new_position[0] <=38   && new_position[1]  >=21.27
//	    		 && new_position[1] <=51  )
//	 {
//    				    		   value[0]=0;//u_h;//u_h*exp(-alpha*(1-(p[0]/h)));
//    				    		  value[1] =0;//u_h*exp(-alpha*(1-(p[1]/h)));
//
//    				     }
//    				    		  else if (p[0]>=-300  && p[0]<=500 && p[1]==0 )
//    				    		  	       {
//  		    			  value[0]=0;//u_h*exp(-alpha*(1-(p[0]/h)));
//  						  value[1] =0;
//  		    		  	       }
//
//
//  else if (  p[1]<=51)
//  		    		  	       {
//  		    			  value[0]=u_h;
//  						  value[1] =0;
//  		    		  	       }
//
//
//  		    	    else {
//  		    		   value[0]=(u_star/k)*log((p[1]+z0)/z0);
//  		    		   value[1] =0;
//  	   }

//     return value;
  //Dynamical cast.
  // return Point<dim>(value);
}

template <int dim>
void AdvectionField<dim>::value_list(const std::vector<Point<dim>> &points,
									 std::vector<Tensor<1, dim>> &values) const
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

#endif /* INCLUDE_ADVECTION_FIELD_HPP_ */
