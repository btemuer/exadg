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
    unsigned int const     cell     = integrator.get_current_cell_index();
    SolutionGradient const gradient = integrator.get_gradient(q);

    Coefficient const      coefficient = kernel->get_coefficient_cell(cell, q);
    SolutionGradient const volume_flux = kernel->get_volume_flux(gradient, coefficient);

    integrator.submit_gradient(volume_flux, q);
  }
}
} // namespace ExaDG