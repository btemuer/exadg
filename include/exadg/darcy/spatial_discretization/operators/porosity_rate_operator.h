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

#ifndef EXADG_DARCY_ALE_MASS_OPERATOR_H
#define EXADG_DARCY_ALE_MASS_OPERATOR_H

#include "exadg/functions_and_boundary_conditions/evaluate_functions.h"
#include "exadg/matrix_free/integrators.h"
#include "exadg/operators/mapping_flags.h"

namespace ExaDG
{
namespace Darcy
{
namespace Operators
{
template<int dim>
struct PorosityRateData
{
  std::shared_ptr<dealii::Function<dim>> initial_porosity_field;
};

template<int dim, typename Number>
class PorosityRate
{
private:
  using IntegratorCell = CellIntegrator<dim, dim, Number>;
  using IntegratorFace = FaceIntegrator<dim, dim, Number>;

  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;

  using point  = dealii::Point<dim, dealii::VectorizedArray<Number>>;
  using scalar = dealii::VectorizedArray<Number>;
  using vector = dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>;

public:
  void
  reinit(dealii::MatrixFree<dim, Number> const & matrix_free,
         PorosityRateData<dim> const &     data_in,
         unsigned int const                      dof_index,
         unsigned int const                      quad_index) const
  {
    data = data_in;
  }

  static MappingFlags
  get_mapping_flags()
  {
    MappingFlags flags;

    flags.cells =
      dealii::update_JxW_values | dealii::update_gradients | dealii::update_quadrature_points;
    flags.inner_faces =
      dealii::update_JxW_values | dealii::update_normal_vectors | dealii::update_quadrature_points;
    flags.boundary_faces =
      dealii::update_JxW_values | dealii::update_normal_vectors | dealii::update_quadrature_points;

    return flags;
  }

  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_flux(vector const & value_m, vector const & value_p, point const & q) const
  {
    AssertThrow(data.initial_porosity_field,
                dealii::ExcMessage("Initial porosity field function not set."));

    auto const initial_porosity =
      FunctionEvaluator<0, dim, Number>::value(data.initial_porosity_field, q, 0.0);

    return initial_porosity * 0.5 * (value_m + value_p);
  }

  inline DEAL_II_ALWAYS_INLINE //
    vector
    get_volume_flux(IntegratorCell & grid_velocity, unsigned int const q) const
  {
    AssertThrow(data.initial_porosity_field,
                dealii::ExcMessage("Initial porosity field function not set."));

    scalar const initial_porosity =
      FunctionEvaluator<0, dim, Number>::value(data.initial_porosity_field,
                                               grid_velocity.quadrature_point(q),
                                               0.0);

    return -(1.0 - initial_porosity) * grid_velocity.get_value(q);
  }

private:
  mutable PorosityRateData<dim> data;
};
} // namespace Operators
} // namespace Darcy
} // namespace ExaDG

#endif // EXADG_DARCY_ALE_MASS_OPERATOR_H
