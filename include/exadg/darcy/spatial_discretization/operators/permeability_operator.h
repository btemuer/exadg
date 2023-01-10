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

#ifndef EXADG_DARCY_HETEROGENEOUS_MASS_OPERATOR_H
#define EXADG_DARCY_HETEROGENEOUS_MASS_OPERATOR_H

#include <exadg/functions_and_boundary_conditions/evaluate_functions.h>
#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/mapping_flags.h>


namespace ExaDG
{
namespace Darcy
{
namespace Operators
{
template<int dim, typename Number>
struct PermeabilityKernelData
{
  std::shared_ptr<dealii::Function<dim>> porosity_field;
  std::shared_ptr<dealii::Function<dim>> inverse_permeability_field;
  Number                                 viscosity;
};

template<int dim, typename Number>
class PermeabilityKernel
{
private:
  using CellIntegratorU = CellIntegrator<dim, dim, Number>;

  using scalar = dealii::VectorizedArray<Number>;
  using vector = dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>;

public:
  void
  reinit(PermeabilityKernelData<dim, Number> const & data_in) const
  {
    data = data_in;
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
    get_volume_flux(CellIntegratorU const & velocity,
                    unsigned int const      q,
                    Number const           time) const
  {
    AssertThrow(data.viscosity > 0.0, dealii::ExcMessage("Problem with the viscosity."));
    AssertThrow(data.porosity_field, dealii::ExcMessage("Porosity field function not set."));
    AssertThrow(data.inverse_permeability_field, dealii::ExcMessage("Inverse permeability field function not set."));

    auto const viscosity = dealii::make_vectorized_array<Number>(data.viscosity);
    auto const porosity  = FunctionEvaluator<0, dim, Number>::value(data.porosity_field,
                                                                   velocity.quadrature_point(q),
                                                                   time);
    auto const inverse_permeability =
      FunctionEvaluator<2, dim, Number>::value_symmetric(data.inverse_permeability_field,
                                                         velocity.quadrature_point(q),
                                                         time);

    return viscosity * porosity * inverse_permeability * velocity.get_value(q);
  }

private:
  mutable PermeabilityKernelData<dim, Number> data;
};
} // namespace Operators

template<int dim, typename Number>
struct PermeabilityOperatorData
{
  unsigned int dof_index_velocity{};
  unsigned int quad_index_velocity{};

  Operators::PermeabilityKernelData<dim, Number> kernel_data;
};

template<int dim, typename Number>
class PermeabilityOperator
{
private:
  using This = PermeabilityOperator<dim, Number>;

  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;

  using scalar = dealii::VectorizedArray<Number>;
  using vector = dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>;

  using Range = std::pair<unsigned int, unsigned int>;

  using CellIntegratorU = CellIntegrator<dim, dim, Number>;

public:
  PermeabilityOperator() = default;

  void
  initialize(dealii::MatrixFree<dim, Number> const &       matrix_free,
             PermeabilityOperatorData<dim, Number> const & data);

  void
  apply(VectorType & dst, VectorType const & src) const;

  void
  apply_add(VectorType & dst, VectorType const & src) const;

private:
  void
  do_cell_integral(CellIntegratorU & velocity) const;

  void
  cell_loop(dealii::MatrixFree<dim, Number> const & matrix_free,
            VectorType &                            dst,
            VectorType const &                      src,
            Range const &                           cell_range) const;

  dealii::MatrixFree<dim, Number> const * matrix_free;

  mutable double time{};

  Operators::PermeabilityKernel<dim, Number> kernel;

  PermeabilityOperatorData<dim, Number> data;
};

} // namespace Darcy
} // namespace ExaDG
#endif // EXADG_DARCY_HETEROGENEOUS_MASS_OPERATOR_H
