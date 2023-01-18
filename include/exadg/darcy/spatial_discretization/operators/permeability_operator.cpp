/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#include <exadg/darcy/spatial_discretization/operators/permeability_operator.h>

namespace ExaDG
{
namespace Darcy
{
template<int dim, typename Number>
void
PermeabilityOperator<dim, Number>::initialize(
  dealii::MatrixFree<dim, Number> const & matrix_free_in,
  PermeabilityOperatorData<dim> const &   data_in)
{
  this->matrix_free = &matrix_free_in;
  this->data        = data_in;

  kernel.reinit(data_in.kernel_data);
}


template<int dim, typename Number>
void
PermeabilityOperator<dim, Number>::apply(PermeabilityOperator::VectorType &       dst,
                                         const PermeabilityOperator::VectorType & src) const
{
  matrix_free->cell_loop(&This::cell_loop, this, dst, src, true);
}

template<int dim, typename Number>
void
PermeabilityOperator<dim, Number>::apply_add(PermeabilityOperator::VectorType &       dst,
                                             PermeabilityOperator::VectorType const & src) const
{
  matrix_free->cell_loop(&This::cell_loop, this, dst, src, false);
}

template<int dim, typename Number>
void
PermeabilityOperator<dim, Number>::cell_loop(dealii::MatrixFree<dim, Number> const & matrix_free,
                                             VectorType &                            dst,
                                             VectorType const &                      src,
                                             Range const & cell_range) const
{
  CellIntegratorU velocity(matrix_free, data.dof_index_velocity, data.quad_index_velocity);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    velocity.reinit(cell);

    velocity.gather_evaluate(src, dealii::EvaluationFlags::values);

    do_cell_integral(velocity);

    velocity.integrate_scatter(dealii::EvaluationFlags::values, dst);
  }
}

template<int dim, typename Number>
void
PermeabilityOperator<dim, Number>::do_cell_integral(CellIntegratorU & velocity) const
{
  for(unsigned int q = 0; q < velocity.n_q_points; ++q)
  {
    velocity.submit_value(
      kernel.get_volume_flux(velocity.get_value(q), velocity.quadrature_point(q), this->time), q);
  }
}

template class PermeabilityOperator<2, float>;
template class PermeabilityOperator<2, double>;

template class PermeabilityOperator<3, float>;
template class PermeabilityOperator<3, double>;

} // namespace Darcy
} // namespace ExaDG