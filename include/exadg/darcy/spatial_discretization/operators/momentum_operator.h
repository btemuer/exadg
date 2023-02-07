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
#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/weak_boundary_conditions.h>
#include <exadg/operators/mass_kernel.h>
#include <exadg/operators/operator_base.h>
#include <exadg/darcy/spatial_discretization/grid_velocity_manager.h>
#include <exadg/darcy/spatial_discretization/operators/grid_convection_operator.h>

namespace ExaDG
{
namespace Darcy
{
template<int dim>
struct MomentumOperatorData : public OperatorBaseData
{
  bool unsteady_problem{false};

  bool ale{false};

  Operators::PermeabilityKernelData<dim> permeability_kernel_data;

  Operators::GridConvectionKernelData ale_momentum_kernel_data;

  bool use_boundary_data{true};

  std::shared_ptr<IncNS::BoundaryDescriptorU<dim> const> bc;
};

template<int dim, typename Number>
class MomentumOperator : public OperatorBase<dim, Number, dim>
{
private:
  using scalar = dealii::VectorizedArray<Number>;
  using vector = dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>;
  using dyadic = dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>;

  using This = MomentumOperator<dim, Number>;
  using Base = OperatorBase<dim, Number, dim>;

  using Base::rhs;
  using Base::rhs_add;

  using Range          = typename Base::Range;
  using VectorType     = typename Base::VectorType;
  using IntegratorCell = typename Base::IntegratorCell;
  using IntegratorFace = typename Base::IntegratorFace;

public:
  using value_type = Number;

  void
  initialize(dealii::MatrixFree<dim, Number> const &           matrix_free,
             dealii::AffineConstraints<Number> const &         affine_constraints,
             MomentumOperatorData<dim> const &                 data,
             std::shared_ptr<GridVelocityManager<dim, Number>> grid_velocity_manager_in);

  MomentumOperatorData<dim> const &
  get_data() const;

  Number
  get_scaling_factor_mass_operator() const;

  void
  set_scaling_factor_mass_operator(Number factor);

  void
  rhs(VectorType & rhs, double evaluation_time) const;

  void
  rhs_add(VectorType & rhs, double evaluation_time) const;

private:
  void
  reinit_cell(unsigned int cell) const override;

  void
  reinit_face(unsigned int face) const override;

  void
  reinit_boundary_face(unsigned int face) const override;

  void
  do_cell_integral(IntegratorCell & integrator) const override;

  void
  do_face_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const override;

  void
  do_face_int_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const override;

  void
  do_boundary_integral(IntegratorFace &                   integrator,
                       const OperatorType &               operator_type,
                       const dealii::types::boundary_id & boundary_id) const override;

  void
  cell_loop_inhom_operator(dealii::MatrixFree<dim, Number> const &       matrix_free,
                           VectorType &                                  dst,
                           VectorType const &                            src,
                           std::pair<unsigned int, unsigned int> const & cell_range) const;

  void
  face_loop_empty(dealii::MatrixFree<dim, Number> const & matrix_free,
                  VectorType &                            dst,
                  VectorType const &                      src,
                  Range const &                           range) const;

  // inhomogeneous operator
  void
  boundary_face_loop_inhom_operator(dealii::MatrixFree<dim, Number> const & matrix_free,
                                    VectorType &                            dst,
                                    VectorType const &                      src,
                                    Range const &                           range) const;

  MomentumOperatorData<dim> operator_data;

  std::shared_ptr<MassKernel<dim, Number>>                      mass_kernel;
  std::shared_ptr<Operators::PermeabilityKernel<dim, Number>>   permeability_kernel;
  std::shared_ptr<Operators::GridConvectionKernel<dim, Number>> grid_convection_kernel;

  std::shared_ptr<GridVelocityManager<dim, Number>> grid_velocity_manager;

  double scaling_factor_mass{1.0};

  mutable double time{};
};

} // namespace Darcy
} // namespace ExaDG

#endif // EXADG_DARCY_MOMENTUM_OPERATOR_H
