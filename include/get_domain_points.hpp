/*
 * GET_DOMAIN_POINTS.hpp
 *
 *  Created on: Nov 27, 2021
 *      Author: heena
 */

#ifndef PROJECT_INCLUDE_GET_DOMAIN_POINTS_HPP_
#define PROJECT_INCLUDE_GET_DOMAIN_POINTS_HPP_

// Deal.ii
#include <deal.II/base/point.h>
#include <deal.II/base/geometry_info.h>

// STL
#include <cmath>
#include <vector>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/mapping_q1.h>

// My headers
#include <config.h>

namespace Coefficients
{
using namespace dealii;

template <int dim>
class UnitCellPointFinder
{
public:
  UnitCellPointFinder();


  virtual Point<dim>
  value(const Point<dim> &p) override;

private:
  const Mapping<dim> &mapping;

  typename DoFHandler<dim>::active_cell_iterator this_global_cell;

};



template <int dim>
Point<dim>
UnitCellPointFinder<dim>::value(const Point<dim> &p)
{
//      Point<dim> unit_cell_point = GeometryInfo<dim>::project_to_unit_cell(p);


//transform_real_to_unit_cell : Map the point p on the real cell to the corresponding point on the unit cell, and return its coordinates.
      Point<dim> unit_cell_point =mapping.transform_real_to_unit_cell(this_global_cell,p);
//   is_inside_unit_cell : Return true if the given point is inside the unit cell of the present space dimension.
          bool inside_unit_cell=GeometryInfo<dim>::is_inside_unit_cell(unit_cell_point);
          if(inside_unit_cell)
        	     return unit_cell_point;

}



} // end namespace Coefficients



#endif /* PROJECT_INCLUDE_GET_DOMAIN_POINTS_HPP_ */
