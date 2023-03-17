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

#include <exadg/grid/grid_utilities.h>
#include <exadg/operators/interior_penalty_parameter.h>
#include <exadg/operators/operator_base.h>
#include <exadg/operators/variable_coefficients.h>

namespace ExaDG
{
namespace Operators
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

  static constexpr unsigned int value_rank = (n_components > 1) ? 1 : 0;
  static constexpr unsigned int coefficient_rank =
    (coupling_coefficient) ? ((n_components > 1) ? 4 : 2) : 0;

  using Value    = dealii::Tensor<value_rank, dim, scalar>;
  using Gradient = dealii::Tensor<value_rank + 1, dim, scalar>;

  using Coefficient = dealii::Tensor<coefficient_rank, dim, scalar>;

  typedef CellIntegrator<dim, n_components, Number> IntegratorCell;
  typedef FaceIntegrator<dim, n_components, Number> IntegratorFace;

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

    return flags;
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
    Gradient
    get_volume_flux(Gradient const & gradient, Coefficient const & coefficient)
  {
    return coefficient * gradient;
  }

  static inline DEAL_II_ALWAYS_INLINE //
    Gradient
    get_gradient_flux(Value const &       value_m,
                      Value const &       value_p,
                      vector const &      normal,
                      Coefficient const & coefficient)
  {
    auto const jump_value  = value_m - value_p;
    auto const jump_tensor = outer_product(jump_value, normal);

    return -0.5 * coefficient * jump_tensor;
  }

  inline DEAL_II_ALWAYS_INLINE //
    Value
    get_value_flux(Value const &       value_m,
                   Value const &       value_p,
                   Gradient const &    gradient_m,
                   Gradient const &    gradient_p,
                   vector const &      normal,
                   Coefficient const & coefficient)
  {
    auto const jump_value  = value_m - value_p;
    auto const jump_tensor = outer_product(jump_value, normal);

    auto const average_gradient = 0.5 * (gradient_m + gradient_p);

    return -(coefficient * (average_gradient + tau * jump_tensor)) * normal;
  }

  void
  calculate_penalty_parameter(dealii::MatrixFree<dim, Number> const & matrix_free,
                              unsigned int const                      dof_index)
  {
    IP::calculate_penalty_parameter<dim, Number>(penalty_parameters, matrix_free, dof_index);
  }

  void
  calculate_coefficients(dealii::MatrixFree<dim, Number> const & matrix_free,
                         unsigned int const                      quad_index)
  {
    coefficients.initialize(matrix_free, quad_index, data.coefficient_function);
  }

  void
  reinit_face(IntegratorFace &   integrator_m,
              IntegratorFace &   integrator_p,
              unsigned int const dof_index) const
  {
    tau = std::max(integrator_m.read_cell_data(penalty_parameters),
                   integrator_p.read_cell_data(penalty_parameters)) *
          IP::get_penalty_factor<dim, Number>(
            degree,
            GridUtilities::get_element_type(
              integrator_m.get_matrix_free().get_dof_handler(dof_index).get_triangulation()),
            data.IP_factor);
  }

  void
  reinit_boundary_face(IntegratorFace & integrator_m, unsigned int const dof_index) const
  {
    tau = integrator_m.read_cell_data(penalty_parameters) *
          IP::get_penalty_factor<dim, Number>(
            degree,
            GridUtilities::get_element_type(
              integrator_m.get_matrix_free().get_dof_handler(dof_index).get_triangulation()),
            data.IP_factor);
  }

  void
  reinit_face_cell_based(dealii::types::boundary_id const boundary_id,
                         IntegratorFace &                 integrator_m,
                         IntegratorFace &                 integrator_p,
                         unsigned int const               dof_index) const
  {
    if(boundary_id == dealii::numbers::internal_face_boundary_id) // internal face
    {
      tau = std::max(integrator_m.read_cell_data(penalty_parameters),
                     integrator_p.read_cell_data(penalty_parameters)) *
            IP::get_penalty_factor<dim, Number>(
              degree,
              GridUtilities::get_element_type(
                integrator_m.get_matrix_free().get_dof_handler(dof_index).get_triangulation()),
              data.IP_factor);
    }
    else // boundary face
    {
      tau = integrator_m.read_cell_data(penalty_parameters) *
            IP::get_penalty_factor<dim, Number>(
              degree,
              GridUtilities::get_element_type(
                integrator_m.get_matrix_free().get_dof_handler(dof_index).get_triangulation()),
              data.IP_factor);
    }
  }

private:
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
} // namespace Operators

template<int dim, typename Number, int n_components = 1, bool coupling_coefficient = false>
struct GeneralizedLaplaceOperatorData : public OperatorBaseData
{
  Operators::GeneralizedLaplaceKernelData<dim, Number, n_components, coupling_coefficient>
    kernel_data;

  // TODO
  // std::shared_ptr<BoundaryDescriptor<dim> const> bc;
};

template<int dim, typename Number, int n_components = 1, bool coupling_coefficient = false>
class GeneralizedLaplaceOperator : public OperatorBase<dim, Number, n_components>
{
private:
  using scalar = dealii::VectorizedArray<Number>;
  using vector = dealii::Tensor<1, dim, scalar>;

  static constexpr unsigned int value_rank = (n_components > 1) ? 1 : 0;
  static constexpr unsigned int coefficient_rank =
    (coupling_coefficient) ? ((n_components > 1) ? 4 : 2) : 0;

  using Value    = dealii::Tensor<value_rank, dim, scalar>;
  using Gradient = dealii::Tensor<value_rank + 1, dim, scalar>;

  using Coefficient = dealii::Tensor<coefficient_rank, dim, scalar>;

  using Base = OperatorBase<dim, Number, n_components>;

  using Range          = typename Base::Range;
  using VectorType     = typename Base::VectorType;
  using IntegratorCell = typename Base::IntegratorCell;
  using IntegratorFace = typename Base::IntegratorFace;

public:
  void
  initialize(
    dealii::MatrixFree<dim, Number> const &   matrix_free,
    dealii::AffineConstraints<Number> const & affine_constraints,
    GeneralizedLaplaceOperatorData<dim, Number, n_components, coupling_coefficient> const & data);

  void
  initialize(
    dealii::MatrixFree<dim, Number> const &   matrix_free,
    dealii::AffineConstraints<Number> const & affine_constraints,
    GeneralizedLaplaceOperatorData<dim, Number, n_components, coupling_coefficient> const & data,
    std::shared_ptr<
      Operators::GeneralizedLaplaceKernel<dim, Number, n_components, coupling_coefficient>>
      generalized_laplace_kernel);

  void
  update();

private:
  void
  reinit_face(unsigned int const face) const;

  void
  reinit_boundary_face(unsigned int const face) const;

  void
  reinit_face_cell_based(unsigned int const               cell,
                         unsigned int const               face,
                         dealii::types::boundary_id const boundary_id) const;

  void
  do_cell_integral(IntegratorCell & integrator) const;

  void
  do_face_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const;

  void
  do_face_int_integral(IntegratorFace & integrator_m,
                       IntegratorFace & integrator_p,
                       bool             revert_int_ext = false) const;

  void
  do_face_ext_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const;

  void
  do_boundary_integral(IntegratorFace &                   integrator,
                       OperatorType const &               operator_type,
                       dealii::types::boundary_id const & boundary_id) const;

  GeneralizedLaplaceOperatorData<dim, Number, n_components, coupling_coefficient> operator_data;

  std::shared_ptr<
    Operators::GeneralizedLaplaceKernel<dim, Number, n_components, coupling_coefficient>>
    kernel;
};
} // namespace ExaDG
#endif /* INCLUDE_EXADG_OPERATORS_GENERALIZED_LAPLACE_OPERATOR_H_ */
