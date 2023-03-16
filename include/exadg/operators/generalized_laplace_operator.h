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

#ifndef INCLUDE_EXADG_OPERATORS_GENERALIZED_LAPLACE_OPERATOR_H_
#define INCLUDE_EXADG_OPERATORS_GENERALIZED_LAPLACE_OPERATOR_H_

#include <exadg/operators/interior_penalty_parameter.h>
#include <exadg/operators/operator_base.h>
#include <exadg/operators/variable_coefficients.h>

namespace ExaDG
{
template<int dim, typename Number, int n_components = 1, bool coupling_coefficient = false>
struct GeneralizedLaplaceKernelData
{
private:
  static constexpr unsigned int coefficient_rank =
    (coupling_coefficient) ? ((n_components > 1) ? 4 : 2) : 0;

  using scalar = dealii::VectorizedArray<Number>;

  using Coefficient = dealii::Tensor<coefficient_rank, dim, scalar>;

  using CoefficientFunction = std::function<Coefficient(unsigned int, unsigned int)>;

public:
  double IP_factor{1.0};

  CoefficientFunction coefficient_function{};
};

template<int dim, typename Number, int n_components = 1, bool coupling_coefficient = false>
class GeneralizedLaplaceKernel
{
private:
  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;

  using scalar = dealii::VectorizedArray<Number>;
  using vector = dealii::Tensor<1, dim, scalar>;

  static constexpr unsigned int solution_rank = (n_components > 1) ? 1 : 0;
  static constexpr unsigned int coefficient_rank =
    (coupling_coefficient) ? ((n_components > 1) ? 4 : 2) : 0;

  using Solution         = dealii::Tensor<solution_rank, dim, scalar>;
  using SolutionGradient = dealii::Tensor<solution_rank + 1, dim, scalar>;

  using Coefficient = dealii::Tensor<coefficient_rank, dim, scalar>;

public:
  template<typename F>
  void
  reinit(
    dealii::MatrixFree<dim, Number> const & matrix_free,
    GeneralizedLaplaceKernelData<dim, Number, n_components, coupling_coefficient> const & data_in,
    unsigned int const                                                                    dof_index,
    unsigned int const quad_index)
  {
    data   = data_in;
    degree = matrix_free.get_dof_handler(dof_index).get_fe().degree;

    calculate_penalty_parameter(matrix_free, dof_index);

    coefficients.initialize(matrix_free, quad_index, data.coefficient_function);
  }

  static IntegratorFlags
  get_integrator_flags()
  {
    IntegratorFlags flags;

    flags.cell_evaluate  = dealii::EvaluationFlags::gradients;
    flags.cell_integrate = dealii::EvaluationFlags::gradients;

    flags.face_evaluate  = dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients;
    flags.face_integrate = dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients;
  }

  static MappingFlags
  get_mapping_flags(bool const compute_interior_face_integrals,
                    bool const compute_boundary_face_integrals)
  {
    MappingFlags flags;

    flags.cells = dealii::update_JxW_values | dealii::update_gradients;
    if(compute_interior_face_integrals)
      flags.inner_faces =
        dealii::update_JxW_values | dealii::update_gradients | dealii::update_normal_vectors;
    if(compute_boundary_face_integrals)
      flags.boundary_faces = dealii::update_JxW_values | dealii::update_gradients |
                             dealii::update_normal_vectors | dealii::update_quadrature_points;

    return flags;
  }

  static inline DEAL_II_ALWAYS_INLINE //
    SolutionGradient
    get_volume_flux(SolutionGradient const & gradient, Coefficient const & coefficient)
  {
    return coefficient * gradient;
  }

  static inline DEAL_II_ALWAYS_INLINE //
    SolutionGradient
    get_gradient_flux(Solution const &    value_m,
                      Solution const &    value_p,
                      vector const &      normal,
                      Coefficient const & coefficient)
  {
    auto const jump_value  = value_m - value_p;
    auto const jump_tensor = outer_product(jump_value, normal);

    return -0.5 * coefficient * jump_tensor;
  }

private:
  void
  calculate_penalty_parameter(dealii::MatrixFree<dim, Number> const & matrix_free,
                              unsigned int const                      dof_index)
  {
    IP::calculate_penalty_parameter<dim, Number>(penalty_parameters, matrix_free, dof_index);
  }

  template<typename F>
  void
  set_coefficients(F const & coefficient_function)
  {
    coefficients.set_cofficients(coefficient_function);
  }

  GeneralizedLaplaceKernelData<dim, Number, n_components, coupling_coefficient> data{};

  unsigned int degree{1};

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