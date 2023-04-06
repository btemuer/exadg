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

#include <exadg/functions_and_boundary_conditions/evaluate_functions.h>
#include <exadg/grid/grid_utilities.h>
#include <exadg/operators/interior_penalty_parameter.h>
#include <exadg/operators/operator_base.h>
#include <exadg/operators/variable_coefficients.h>
#include <exadg/poisson/user_interface/boundary_descriptor.h>

namespace ExaDG
{
namespace GeneralizedLaplaceOperator
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

  using coefficient_function_type =
    std::function<dealii::Tensor<coefficient_rank, dim, scalar>(unsigned int, unsigned int)>;

public:
  double IP_factor{1.0};

  coefficient_function_type coefficient_function{};
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

  using value_type    = dealii::Tensor<value_rank, dim, scalar>;
  using gradient_type = dealii::Tensor<value_rank + 1, dim, scalar>;

  using coefficient_type = dealii::Tensor<coefficient_rank, dim, scalar>;

  using IntegratorCell = CellIntegrator<dim, n_components, Number>;
  using IntegratorFace = FaceIntegrator<dim, n_components, Number>;

public:
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
    gradient_type
    get_volume_flux(gradient_type const & gradient, coefficient_type const & coefficient)
  {
    return coeff_mult(coefficient, gradient);
  }

  static inline DEAL_II_ALWAYS_INLINE //
    gradient_type
    get_gradient_flux(value_type const &       value_m,
                      value_type const &       value_p,
                      vector const &           normal,
                      coefficient_type const & coefficient)
  {
    value_type const    jump_value  = value_m - value_p;
    gradient_type const jump_tensor = outer_product(jump_value, normal);

    return -0.5 * coeff_mult(coefficient, jump_tensor);
  }

  inline DEAL_II_ALWAYS_INLINE //
    value_type
    get_value_flux(gradient_type const &    gradient_m,
                   gradient_type const &    gradient_p,
                   value_type const &       value_m,
                   value_type const &       value_p,
                   vector const &           normal,
                   coefficient_type const & coefficient)
  {
    value_type const    jump_value  = value_m - value_p;
    gradient_type const jump_tensor = outer_product(jump_value, normal);

    gradient_type const average_gradient = 0.5 * (gradient_m + gradient_p);

    return -coeff_mult(coefficient, (average_gradient + tau * jump_tensor)) * normal;
  }

  inline DEAL_II_ALWAYS_INLINE //
    value_type
    get_value_flux(value_type const &       coeff_times_normal_gradient_m,
                   value_type const &       coeff_times_normal_gradient_p,
                   value_type const &       value_m,
                   value_type const &       value_p,
                   vector const &           normal,
                   coefficient_type const & coefficient)
  {
    value_type const    jump_value  = value_m - value_p;
    gradient_type const jump_tensor = outer_product(jump_value, normal);

    value_type const average_coeff_times_normal_gradient =
      0.5 * (coeff_times_normal_gradient_m + coeff_times_normal_gradient_p);

    return -(average_coeff_times_normal_gradient +
             value_type(coeff_mult(coefficient, (tau * jump_tensor)) * normal));
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

  coefficient_type
  get_coefficient_cell(unsigned int const cell, unsigned int const q)
  {
    return coefficients.get_coefficient_cell(cell, q);
  }

  coefficient_type
  get_coefficient_face(unsigned int const face, unsigned int const q)
  {
    return coefficients.get_coefficient_face(face, q);
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
  template<typename T>
  static inline DEAL_II_ALWAYS_INLINE //
    T
    coeff_mult(coefficient_type const & coeff, T const & x)
  {
    if constexpr(coefficient_rank == 4)
      return dealii::double_contract<2, 0, 3, 1>(coeff, x);
    else
      return coeff * x;
  }

  GeneralizedLaplaceKernelData<dim, Number, n_components, coupling_coefficient> data{};

  unsigned int degree{1};

  mutable scalar tau{0.0};

  dealii::AlignedVector<scalar>                  penalty_parameters{};
  mutable VariableCoefficients<coefficient_type> coefficients{};
};
} // namespace Operators

namespace Boundary
{
template<int dim, typename Number, int n_components, bool coupling_coefficient>
struct WeakBoundaryConditions
{
  using scalar = dealii::VectorizedArray<Number>;
  using vector = dealii::Tensor<1, dim, scalar>;

  static constexpr unsigned int value_rank = (n_components > 1) ? 1 : 0;
  static constexpr unsigned int coefficient_rank =
    (coupling_coefficient) ? ((n_components > 1) ? 4 : 2) : 0;

  using value_type    = dealii::Tensor<value_rank, dim, scalar>;
  using gradient_type = dealii::Tensor<value_rank + 1, dim, scalar>;

  using coefficient_type = dealii::Tensor<coefficient_rank, dim, scalar>;

  static inline DEAL_II_ALWAYS_INLINE //
    value_type
    calculate_interior_value(unsigned int const                                q,
                             FaceIntegrator<dim, n_components, Number> const & integrator,
                             OperatorType const &                              operator_type)
  {
    if(operator_type == OperatorType::full || operator_type == OperatorType::homogeneous)
      return integrator.get_value(q);
    else if(operator_type == OperatorType::inhomogeneous)
      return value_type{};
    else
      AssertThrow(false, dealii::ExcMessage("Specified OperatorType is not implemented!"));

    return value_type{};
  }

  static inline DEAL_II_ALWAYS_INLINE //
    value_type
    calculate_exterior_value(
      value_type const &                                                  value_m,
      unsigned int const                                                  q,
      FaceIntegrator<dim, n_components, Number> const &                   integrator,
      OperatorType const &                                                operator_type,
      Poisson::BoundaryType const &                                       boundary_type,
      dealii::types::boundary_id const                                    boundary_id,
      std::shared_ptr<Poisson::BoundaryDescriptor<value_rank, dim> const> boundary_descriptor,
      double const &                                                      time)
  {
    if(boundary_type == Poisson::BoundaryType::Neumann)
      return value_m;

    if(operator_type == OperatorType::full || operator_type == OperatorType::inhomogeneous)
    {
      value_type g{};

      if(boundary_type == Poisson::BoundaryType::Dirichlet)
      {
        auto const bc       = boundary_descriptor->dirichlet_bc.find(boundary_id)->second;
        auto const q_points = integrator.quadrature_point(q);

        g = FunctionEvaluator<value_rank, dim, Number>::value(bc, q_points, time);
      }
      else
        AssertThrow(false,
                    dealii::ExcMessage("Boundary type of face is invalid or not implemented."));

      return -value_m + value_type(2.0 * g);
    }
    else if(operator_type == OperatorType::homogeneous)
      return -value_m;
    else
      AssertThrow(false, dealii::ExcNotImplemented());

    return value_type{};
  }

  static inline DEAL_II_ALWAYS_INLINE //
    value_type
    calculate_interior_coeff_times_normal_gradient(
      unsigned int const                                q,
      FaceIntegrator<dim, n_components, Number> const & integrator,
      OperatorType const &                              operator_type,
      coefficient_type const &                          coefficient)
  {
    if(operator_type == OperatorType::full || operator_type == OperatorType::homogeneous)
      return coeff_mult(coefficient, integrator.get_gradient(q)) * integrator.get_normal_vector(q);
    else if(operator_type == OperatorType::inhomogeneous)
      return value_type{};
    else
    {
      AssertThrow(false, dealii::ExcNotImplemented());
      return value_type{};
    }
  }

  static inline DEAL_II_ALWAYS_INLINE //
    value_type
    calculate_exterior_coeff_times_normal_gradient(
      value_type const &                                coeff_times_normal_gradient_m,
      unsigned int const                                q,
      FaceIntegrator<dim, n_components, Number> const & integrator,
      OperatorType const &                              operator_type,
      Poisson::BoundaryType const &                     boundary_type,
      dealii::types::boundary_id const                  boundary_id,
      std::shared_ptr<Poisson::BoundaryDescriptor<value_rank, dim> const> boundary_descriptor,
      double const &                                                      time)
  {
    if(boundary_type == Poisson::BoundaryType::Dirichlet)
      return coeff_times_normal_gradient_m;

    if(boundary_type == Poisson::BoundaryType::Neumann)
    {
      if(operator_type == OperatorType::full || operator_type == OperatorType::inhomogeneous)
      {
        auto const bc       = boundary_descriptor->neumann_bc.find(boundary_id)->second;
        auto const q_points = integrator.quadrature_point(q);

        auto const h = FunctionEvaluator<value_rank, dim, Number>::value(bc, q_points, time);

        return coeff_times_normal_gradient_m + value_type(2.0 * h);
      }
      else if(operator_type == OperatorType::homogeneous)
        return -coeff_times_normal_gradient_m;
      else
        AssertThrow(false, dealii::ExcNotImplemented());
    }

    AssertThrow(false, dealii::ExcMessage("Boundary type of face is invalid or not implemented."));

    return value_type{};
  }

private:
  template<typename T>
  static inline DEAL_II_ALWAYS_INLINE //
    T
    coeff_mult(coefficient_type const & coeff, T const & x)
  {
    if constexpr(coefficient_rank == 4)
      return dealii::double_contract<2, 0, 3, 1>(coeff, x);
    else
      return coeff * x;
  }
};
} // namespace Boundary

template<int dim, typename Number, int n_components = 1, bool coupling_coefficient = false>
struct GeneralizedLaplaceOperatorData : public OperatorBaseData
{
  static constexpr unsigned int value_rank = (n_components > 1) ? 1 : 0;

  Operators::GeneralizedLaplaceKernelData<dim, Number, n_components, coupling_coefficient>
    kernel_data{};

  std::shared_ptr<Poisson::BoundaryDescriptor<value_rank, dim> const> bc{};
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

  using value_type    = dealii::Tensor<value_rank, dim, scalar>;
  using gradient_type = dealii::Tensor<value_rank + 1, dim, scalar>;

  using coefficient_type = dealii::Tensor<coefficient_rank, dim, scalar>;

  using Base = OperatorBase<dim, Number, n_components>;

  using Range          = typename Base::Range;
  using VectorType     = typename Base::VectorType;
  using IntegratorCell = typename Base::IntegratorCell;
  using IntegratorFace = typename Base::IntegratorFace;

  using BC = Boundary::WeakBoundaryConditions<dim, Number, n_components, coupling_coefficient>;

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
  reinit_face(unsigned int face) const override;

  void
  reinit_boundary_face(unsigned int face) const override;

  void
  reinit_face_cell_based(unsigned int               cell,
                         unsigned int               face,
                         dealii::types::boundary_id boundary_id) const override;

  void
  do_cell_integral(IntegratorCell & integrator) const override;

  void
  do_face_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const override;

  void
  do_face_int_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const override;

  void
  do_face_ext_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const override;

  void
  do_boundary_integral(IntegratorFace &                   integrator,
                       OperatorType const &               operator_type,
                       dealii::types::boundary_id const & boundary_id) const override;

  GeneralizedLaplaceOperatorData<dim, Number, n_components, coupling_coefficient> operator_data;

  std::shared_ptr<
    Operators::GeneralizedLaplaceKernel<dim, Number, n_components, coupling_coefficient>>
    kernel;
};
} // namespace GeneralizedLaplaceOperator
} // namespace ExaDG
#endif /* INCLUDE_EXADG_OPERATORS_GENERALIZED_LAPLACE_OPERATOR_H_ */
