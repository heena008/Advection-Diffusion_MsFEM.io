/*
 * solution_gaussian.hpp
 *
 *  Created on: Jun 24, 2020
 *      Author: heena
 */

#ifndef DIFFUSION_PROBLEM_INCLUDE_SOLUTION_GAUSSIAN_HPP_
#define DIFFUSION_PROBLEM_INCLUDE_SOLUTION_GAUSSIAN_HPP_

// Deal.ii
#include <deal.II/base/function.h>

// STL
#include <cmath>
#include <fstream>

using namespace dealii;

template <int dim> class SolutionBase
{
protected:
  static const unsigned int n_mu = 2;
  static const Point<dim> mu[n_mu];
};

template <>
const Point<2> SolutionBase<2>::mu[SolutionBase<2>::n_mu] = {Point<2>(1. / 3, 0.5), Point<2>(2. / 3, 0.5)};

template <int dim>
class Solution : public Function<dim>, protected SolutionBase<dim>
{
public:
  virtual double value(const Point<dim> &p,
                       const unsigned int /*component*/ = 0) const override;

  virtual Tensor<1, dim>
  gradient(const Point<dim> &p,
           const unsigned int component = 0) const override;
};

template <int dim>
double Solution<dim>::value(const Point<dim> &p, const unsigned int) const
{
  double return_value = 0;
  {
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
    return_value = sum / 2 * std::sqrt(4 * numbers::PI * numbers::PI * det_M);
  }
  return return_value;
}

template <int dim>
Tensor<1, dim> Solution<dim>::gradient(const Point<dim> &p,
                                       const unsigned int) const
   {
  Tensor<1, dim> return_value;

  {
    double sum = 0;
    Tensor<2, 2> M;
    for (unsigned int d = 0; d < dim; ++d) {
      M[d][d] = 0.03;
    }

    double det_M = determinant(M);
    for (unsigned int i = 0; i < this->n_mu; ++i) {
      const Tensor<1, dim> x_minus_mui = p - this->mu[i];
      sum += std::exp(-0.5 * (x_minus_mui)*invert(M) * (x_minus_mui));

      return_value = (-invert(M) * x_minus_mui) * sum / 2 *
                     std::sqrt(4 * numbers::PI * numbers::PI * det_M);
    }
  }
  return return_value;
}

#endif /* DIFFUSION_PROBLEM_INCLUDE_SOLUTION_GAUSSIAN_HPP_ */
