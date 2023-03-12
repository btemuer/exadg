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

#ifndef INCLUDE_EXADG_OPERATORS_GENERALIZED_LAPLACE_OPERATOR_H_
#define INCLUDE_EXADG_OPERATORS_GENERALIZED_LAPLACE_OPERATOR_H_

#include <exadg/operators/interior_penalty_parameter.h>
#include <exadg/operators/operator_base.h>
#include <exadg/operators/variable_coefficients.h>

namespace ExaDG
{
struct LaplaceKernelData
{
  double IP_factor{1.0};
};

template<int dim, typename Number, int n_components = 1, bool coupling_coefficient = false>
class GeneralizedLaplaceKernel
{
private:
  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;

  using scalar = dealii::VectorizedArray<Number>;

  static constexpr unsigned int coefficient_rank =
    (coupling_coefficient) ? ((n_components == 1) ? 2 : 4) : 0;

  using Coefficient = dealii::Tensor<coefficient_rank, dim, scalar>;

public:
  void
  reinit(dealii::MatrixFree<dim, Number> const & matrix_free,
         LaplaceKernelData const &               data_in,
         unsigned int const                      dof_index,
         unsigned int const                      quad_index)
  {
    data   = data_in;
    degree = matrix_free.get_dof_handler(dof_index).get_fe().degree;
    coefficients.initialize(matrix_free, quad_index, 0.0);
  }

private:
  void
  calculate_penalty_parameter(dealii::MatrixFree<dim, Number> const & matrix_free,
                              unsigned int const                      dof_index)
  {
    IP::calculate_penalty_parameter<dim, Number>(penalty_parameters, matrix_free, dof_index);
  }

  void
  set_coefficients()
  {
    coefficients.set_coefficient([]() {});
  }

  LaplaceKernelData data{};
  unsigned int      degree{1};

  mutable scalar tau{0.0};

  dealii::AlignedVector<scalar>             penalty_parameters{};
  mutable VariableCoefficients<Coefficient> coefficients{};
};
template<int dim, typename Number, int n_components, bool coupling_coefficient>
class GeneralizedLaplaceOperator : public OperatorBase<dim, Number, n_components>
{
private:
  using Base = OperatorBase<dim, Number, n_components>;

  using IntegratorCell = typename Base::IntegratorCell;
  using IntegratorFace = typename Base::IntegratorFace;

  using VectorType = typename Base::VectorType;

public:
};
} // namespace ExaDG
#endif /* INCLUDE_EXADG_OPERATORS_GENERALIZED_LAPLACE_OPERATOR_H_ */