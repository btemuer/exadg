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

#ifndef EXADG_DARCY_PERMEABILITY_OPERATOR_H
#define EXADG_DARCY_PERMEABILITY_OPERATOR_H

#include <exadg/functions_and_boundary_conditions/evaluate_functions.h>
#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/integrator_flags.h>
#include <exadg/operators/mapping_flags.h>


namespace ExaDG
{
namespace Darcy
{
namespace Operators
{
template<int dim>
struct PermeabilityKernelData
{
  std::shared_ptr<dealii::Function<dim>> porosity_field;
  std::shared_ptr<dealii::Function<dim>> inverse_permeability_field;
  double                                 viscosity;
};

template<int dim, typename Number>
class PermeabilityKernel
{
private:
  using CellIntegratorU = CellIntegrator<dim, dim, Number>;

  using scalar = dealii::VectorizedArray<Number>;
  using vector = dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>;
  using dyadic = dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>;
  using point  = dealii::Point<dim, dealii::VectorizedArray<Number>>;

public:
  void
  reinit(PermeabilityKernelData<dim> const & data_in) const
  {
    data = data_in;
  }

  IntegratorFlags
  get_integrator_flags() const
  {
    IntegratorFlags flags;

    flags.cell_evaluate  = dealii::EvaluationFlags::values;
    flags.cell_integrate = dealii::EvaluationFlags::values;

    return flags;
  }

  static MappingFlags
  get_mapping_flags()
  {
    MappingFlags flags;

    flags.cells = dealii::update_JxW_values | dealii::update_quadrature_points;

    // no face integrals

    return flags;
  }

  /*
   * Volume flux, i.e., the term occurring in the volume integral
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    get_volume_flux(vector const & velocity_value, point const & q_point) const
  {
    AssertThrow(data.viscosity > 0.0, dealii::ExcMessage("Problem with the viscosity."));
    AssertThrow(data.porosity_field, dealii::ExcMessage("Porosity field function not set."));
    AssertThrow(data.inverse_permeability_field,
                dealii::ExcMessage("Inverse permeability field function not set."));

    scalar const viscosity = dealii::make_vectorized_array<Number>(data.viscosity);
    scalar const porosity =
      FunctionEvaluator<0, dim, Number>::value(data.porosity_field, q_point, 0.0);
    dyadic const inverse_permeability =
      FunctionEvaluator<2, dim, Number>::value_symmetric(data.inverse_permeability_field,
                                                         q_point,
                                                         0.0);

    return viscosity * porosity * inverse_permeability * velocity_value;
  }

private:
  mutable PermeabilityKernelData<dim> data;
};
} // namespace Operators

} // namespace Darcy
} // namespace ExaDG
#endif // EXADG_DARCY_PERMEABILITY_OPERATOR_H
