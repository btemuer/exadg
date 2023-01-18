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
#include "momentum_operator.h"

namespace ExaDG
{
namespace Darcy
{
template<int dim, typename Number>
void
MomentumOperator<dim, Number>::initialize(
  dealii::MatrixFree<dim, Number> const &   matrix_free,
  dealii::AffineConstraints<Number> const & affine_constraints,
  MomentumOperatorData<dim> const &         data)
{
  operator_data = data;

  Base::reinit(matrix_free, affine_constraints, data);

  // Create new objects and initialize kernels

  // Permeability operator
  {
    this->permeability_kernel = std::make_shared<Operators::PermeabilityKernel<dim, Number>>();
    this->permeability_kernel->reinit(data.permeability_kernel_data);
    this->integrator_flags =
      this->integrator_flags | this->permeability_kernel->get_integrator_flags();
  }

  // Mass operator
  {
    if(operator_data.unsteady_problem)
    {
      this->mass_kernel      = std::make_shared<MassKernel<dim, Number>>();
      this->integrator_flags = this->integrator_flags | this->mass_kernel->get_integrator_flags();
    }
  }
}

template<int dim, typename Number>
MomentumOperatorData<dim> const &
MomentumOperator<dim, Number>::get_data() const
{
  return operator_data;
}

template<int dim, typename Number>
Number
MomentumOperator<dim, Number>::get_scaling_factor_mass_operator() const
{
  return this->scaling_factor_mass;
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::set_scaling_factor_mass_operator(Number const factor)
{
  this->scaling_factor_mass = factor;
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::do_cell_integral(IntegratorCell & integrator) const
{
  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    vector value = integrator.get_value(q);

    vector value_flux = permeability_kernel->get_volume_flux(value, integrator.quadrature_point(q), this->time);

    if(operator_data.unsteady_problem)
    {
      value_flux += mass_kernel->get_volume_flux(scaling_factor_mass, value);
    }

    integrator.submit_value(value_flux, q);
  }
}

template class MomentumOperator<2, float>;
template class MomentumOperator<2, double>;

template class MomentumOperator<3, float>;
template class MomentumOperator<3, double>;
} // namespace Darcy
} // namespace ExaDG