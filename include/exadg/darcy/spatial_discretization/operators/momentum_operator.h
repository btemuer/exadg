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

#ifndef EXADG_DARCY_MOMENTUM_OPERATOR_H
#define EXADG_DARCY_MOMENTUM_OPERATOR_H

#include <exadg/darcy/spatial_discretization/operators/permeability_operator.h>
#include <exadg/operators/mass_kernel.h>
#include <exadg/operators/operator_base.h>

namespace ExaDG
{
namespace Darcy
{
template<int dim>
struct MomentumOperatorData : public OperatorBaseData
{
  bool unsteady_problem{false};

  Operators::PermeabilityKernelData<dim> permeability_kernel_data;
};

template<int dim, typename Number>
class MomentumOperator : public OperatorBase<dim, Number, dim>
{
private:
  using scalar = dealii::VectorizedArray<Number>;
  using vector = dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>;

  using Base = OperatorBase<dim, Number, dim>;

  using VectorType     = typename Base::VectorType;
  using IntegratorCell = typename Base::IntegratorCell;
  using IntegratorFace = typename Base::IntegratorFace;

public:
  using value_type = Number;

  void
  initialize(dealii::MatrixFree<dim, Number> const &   matrix_free,
             dealii::AffineConstraints<Number> const & affine_constraints,
             MomentumOperatorData<dim> const &         data);

  MomentumOperatorData<dim> const &
  get_data() const;

  Number
  get_scaling_factor_mass_operator() const;

  void
  set_scaling_factor_mass_operator(Number const factor);

private:
  void
  do_cell_integral(IntegratorCell & integrator) const;

  MomentumOperatorData<dim> operator_data;

  std::shared_ptr<MassKernel<dim, Number>>                    mass_kernel;
  std::shared_ptr<Operators::PermeabilityKernel<dim, Number>> permeability_kernel;

  double scaling_factor_mass{1.0};
};

} // namespace Darcy
} // namespace ExaDG

#endif // EXADG_DARCY_MOMENTUM_OPERATOR_H
