/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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

#include <exadg/operators/generalized_laplace_operator.h>

namespace ExaDG
{
template<int dim, typename Number, int n_components, bool coupling_coefficient>
void
GeneralizedLaplaceOperator<dim, Number, n_components, coupling_coefficient>::do_cell_integral(
  IntegratorCell & integrator) const
{
  for(unsigned int q = 0; q < integrator.n_q; ++q)
  {
    unsigned int const cell     = integrator.get_current_cell_index();
    Gradient const     gradient = integrator.get_gradient(q);

    Coefficient const coefficient = kernel->get_coefficient_cell(cell, q);
    Gradient const    volume_flux = kernel->get_volume_flux(gradient, coefficient);

    integrator.submit_gradient(volume_flux, q);
  }
}

template<int dim, typename Number, int n_components, bool coupling_coefficient>
void
GeneralizedLaplaceOperator<dim, Number, n_components, coupling_coefficient>::do_face_integral(
  IntegratorFace & integrator_m,
  IntegratorFace & integrator_p) const
{
  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    unsigned int const cell = integrator_m.get_current_cell_index();

    Value const value_m = integrator_m.get_value(q);
    Value const value_p = integrator_p.get_value(q);

    Gradient const gradient_m = integrator_m.get_gradient(q);
    Gradient const gradient_p = integrator_p.get_gradient(q);

    vector const normal = integrator_m.get_normal_vector(q);

    Coefficient const coefficient = kernel->get_coefficient(cell, q);

    Gradient const gradient_flux = kernel->get_gradient_flux(value_m, value_p, normal, coefficient);

    Value const value_flux =
      kernel->get_value_flux(gradient_m, gradient_p, value_m, value_p, normal, coefficient);

    integrator_m.submit_gradient(gradient_flux, q);
    integrator_p.submit_gradient(gradient_flux, q);

    integrator_m.submit_value(value_flux, q);
    integrator_p.submit_value(-value_flux, q); // - sign since n⁺ = -n⁻
  }
}
} // namespace ExaDG