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
namespace GeneralizedLaplace
{
template<typename T, int coefficient_rank, int dim, typename Number>
static inline DEAL_II_ALWAYS_INLINE //
  T
  coeff_mult(dealii::Tensor<coefficient_rank, dim, Number> const & coeff, T const & x)
{
  if constexpr(coefficient_rank == 4)
    return dealii::double_contract<2, 0, 3, 1>(coeff, x);
  else
    return coeff * x;
}

namespace Operators
{
template<int dim>
struct KernelData
{
  double IP_factor{1.0};

  std::shared_ptr<dealii::Function<dim>> coefficient_function{};
};

template<int dim, typename Number, int n_components = 1, bool coupling_coefficient = false>
class Kernel
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
  reinit(dealii::MatrixFree<dim, Number> const & matrix_free,
         KernelData<dim> const &                 data_in,
         unsigned int const                      dof_index,
         unsigned int const                      quad_index,
         bool const                              use_cell_based_face_loops)
  {
    data   = data_in;
    degree = matrix_free.get_dof_handler(dof_index).get_fe().degree;

    calculate_penalty_parameter(matrix_free, dof_index);

    reinit_coefficients(matrix_free, quad_index, use_cell_based_face_loops);
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

    flags.cells =
      dealii::update_JxW_values | dealii::update_gradients | dealii::update_quadrature_points;
    if(compute_interior_face_integrals)
      flags.inner_faces = dealii::update_JxW_values | dealii::update_gradients |
                          dealii::update_normal_vectors | dealii::update_quadrature_points;
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
    calculate_gradient_flux(value_type const &       value_m,
                            value_type const &       value_p,
                            vector const &           normal,
                            coefficient_type const & coefficient)
  {
    value_type const    jump_value  = value_m - value_p;
    gradient_type const jump_tensor = outer_product(jump_value, normal);

    return -0.5 * coeff_mult(coefficient, jump_tensor);
  }

  static inline DEAL_II_ALWAYS_INLINE //
    value_type
    calculate_normal_derivative_flux(value_type const &       value_m,
                                     value_type const &       value_p,
                                     coefficient_type const & coefficient)
  {
    AssertThrow(not coupling_coefficient,
                dealii::ExcMessage("Normal derivative flux only makes"
                                   "sense with non-coupling coefficients."));

    value_type const jump_value = value_m - value_p;

    return -0.5 * coefficient * jump_value;
  }

  inline DEAL_II_ALWAYS_INLINE //
    value_type
    calculate_value_flux(gradient_type const &    gradient_m,
                         gradient_type const &    gradient_p,
                         value_type const &       value_m,
                         value_type const &       value_p,
                         vector const &           normal,
                         coefficient_type const & coefficient)
  {
    value_type const    jump_value  = value_m - value_p;
    gradient_type const jump_tensor = outer_product(jump_value, normal);

    gradient_type const average_gradient = 0.5 * (gradient_m + gradient_p);

    return -coeff_mult(coefficient, (average_gradient - tau * jump_tensor)) * normal;
  }

  inline DEAL_II_ALWAYS_INLINE //
    value_type
    calculate_value_flux(value_type const &       coeff_times_gradient_times_normal_m,
                         value_type const &       coeff_times_gradient_times_normal_p,
                         value_type const &       value_m,
                         value_type const &       value_p,
                         vector const &           normal,
                         coefficient_type const & coefficient)
  {
    value_type const    jump_value  = value_m - value_p;
    gradient_type const jump_tensor = outer_product(jump_value, normal);

    value_type const average_coeff_times_normal_gradient =
      0.5 * (coeff_times_gradient_times_normal_m + coeff_times_gradient_times_normal_p);

    return -(average_coeff_times_normal_gradient -
             value_type(coeff_mult(coefficient, (tau * jump_tensor)) * normal));
  }

  void
  calculate_penalty_parameter(dealii::MatrixFree<dim, Number> const & matrix_free,
                              unsigned int const                      dof_index)
  {
    IP::calculate_penalty_parameter<dim, Number>(penalty_parameters, matrix_free, dof_index);
  }

  void
  reinit_coefficients(dealii::MatrixFree<dim, Number> const & matrix_free,
                      unsigned int const                      quad_index,
                      bool const                              use_cell_based_face_loops)
  {
    auto const cell_coefficient_function =
      make_cell_coefficient_function(matrix_free, quad_index, {});

    auto const face_coefficient_function =
      make_face_coefficient_function(matrix_free, quad_index, {});

    auto const cell_based_face_coefficient_function = make_cell_based_face_coefficient_function(
      matrix_free, quad_index, {}, use_cell_based_face_loops);

    coefficients.initialize(matrix_free,
                            quad_index,
                            cell_coefficient_function,
                            face_coefficient_function,
                            {} /* neighbor coefficients not needed */,
                            cell_based_face_coefficient_function);
  }

  void
  update_coefficients(dealii::MatrixFree<dim, Number> const & matrix_free,
                      unsigned int const                      quad_index,
                      double const                            time,
                      bool const                              use_cell_based_face_loops = false)
  {
    auto const cell_coefficient_function =
      make_cell_coefficient_function(matrix_free, quad_index, time);

    auto const face_coefficient_function =
      make_face_coefficient_function(matrix_free, quad_index, time);

    auto const cell_based_face_coefficient_function = make_cell_based_face_coefficient_function(
      matrix_free, quad_index, time, use_cell_based_face_loops);

    coefficients.set_coefficients(cell_coefficient_function,
                                  face_coefficient_function,
                                  {} /* neighbor coefficients not needed */,
                                  cell_based_face_coefficient_function);
  }

  inline DEAL_II_ALWAYS_INLINE //
    auto
    make_cell_coefficient_function(dealii::MatrixFree<dim, Number> const & matrix_free,
                                   unsigned int const                      quad_index,
                                   double const                            time) const
  {
    return [&](unsigned int const cell, unsigned int const q) {
      IntegratorCell integrator(matrix_free, {}, quad_index);
      integrator.reinit(cell);
      return FunctionEvaluator<coefficient_rank, dim, Number>::value(
        this->data.coefficient_function, integrator.quadrature_point(q), time);
    };
  }

  inline DEAL_II_ALWAYS_INLINE //
    auto
    make_face_coefficient_function(dealii::MatrixFree<dim, Number> const & matrix_free,
                                   unsigned int const                      quad_index,
                                   double const                            time) const
  {
    return [&](unsigned int const face, unsigned int const q) {
      IntegratorFace integrator(matrix_free, true /* work like an interior face */, {}, quad_index);
      integrator.reinit(face);
      return FunctionEvaluator<coefficient_rank, dim, Number>::value(
        this->data.coefficient_function, integrator.quadrature_point(q), time);
    };
  }

  inline DEAL_II_ALWAYS_INLINE //
    auto
    make_cell_based_face_coefficient_function(dealii::MatrixFree<dim, Number> const & matrix_free,
                                              unsigned int const                      quad_index,
                                              double const                            time,
                                              bool const use_cell_based_face_loops) const
  {
    return std::invoke([&]() -> std::function<coefficient_type(
                               unsigned int const, unsigned int const, unsigned int const)> {
      if(use_cell_based_face_loops)
        return [&](unsigned int const cell, unsigned int const face, unsigned int const q) {
          IntegratorFace integrator(matrix_free,
                                    true /* work like an interior face */,
                                    {},
                                    quad_index);
          integrator.reinit(cell, face);
          return FunctionEvaluator<coefficient_rank, dim, Number>::value(
            this->data.coefficient_function, integrator.quadrature_point(q), time);
        };
      else
        return {};
    });
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

  coefficient_type
  get_coefficient_face_cell_based(unsigned int const face, unsigned int const q)
  {
    return coefficients.get_coefficient_face_cell_based(face, q);
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
  KernelData<dim> data{};

  unsigned int degree{1};

  mutable scalar tau{0.0};

  dealii::AlignedVector<scalar>                  penalty_parameters{};
  mutable VariableCoefficients<coefficient_type> coefficients{};
};
} // namespace Operators

namespace Boundary
{
enum class BoundaryType
{
  Undefined,
  Dirichlet,
  Neumann
};

template<int dim>
class BoundaryDescriptor
{
private:
  using bc_map = std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>;

public:
  BoundaryDescriptor(bc_map const & dirichlet_bc, bc_map const & neumann_bc)
    : dirichlet_bc(dirichlet_bc), neumann_bc(neumann_bc){};

  inline DEAL_II_ALWAYS_INLINE //
    BoundaryType
    get_boundary_type(dealii::types::boundary_id const & boundary_id) const
  {
    if(this->dirichlet_bc.find(boundary_id) != this->dirichlet_bc.end())
      return BoundaryType::Dirichlet;
    else if(this->neumann_bc.find(boundary_id) != this->neumann_bc.end())
      return BoundaryType::Neumann;

    AssertThrow(false, dealii::ExcMessage("Boundary type of face is invalid or not implemented."));

    return BoundaryType::Undefined;
  }

  bc_map const dirichlet_bc;
  bc_map const neumann_bc;
};

template<int dim, typename T>
std::shared_ptr<BoundaryDescriptor<dim>>
create_laplace_boundary_descriptor(T module_bc_descriptor)
{
  return std::make_shared<BoundaryDescriptor<dim>>(module_bc_descriptor->dirichlet_bc,
                                                   module_bc_descriptor->neumann_bc);
}

template<int dim, typename Number, int n_components, bool coupling_coefficient>
struct WeakBoundaryConditions
{
  using scalar = dealii::VectorizedArray<Number>;
  using vector = dealii::Tensor<1, dim, scalar>;

  static constexpr unsigned int value_rank = (n_components > 1) ? 1 : 0;
  static constexpr unsigned int coefficient_rank =
    (coupling_coefficient) ? ((n_components > 1) ? 4 : 2) : 0;

  using value_type = dealii::Tensor<value_rank, dim, scalar>;

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
    calculate_exterior_value(value_type const &                                value_m,
                             unsigned int const                                q,
                             FaceIntegrator<dim, n_components, Number> const & integrator,
                             OperatorType const &                              operator_type,
                             BoundaryType const &                              boundary_type,
                             dealii::types::boundary_id const                  boundary_id,
                             std::shared_ptr<BoundaryDescriptor<dim> const>    boundary_descriptor,
                             double const                                      time)
  {
    if(boundary_type == BoundaryType::Neumann)
      return value_m;

    if(operator_type == OperatorType::full || operator_type == OperatorType::inhomogeneous)
    {
      value_type g{};

      if(boundary_type == BoundaryType::Dirichlet)
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
    calculate_interior_coeff_times_gradient_times_normal(
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
    calculate_exterior_coeff_times_gradient_times_normal(
      value_type const &                                coeff_times_gradient_times_normal_m,
      unsigned int const                                q,
      FaceIntegrator<dim, n_components, Number> const & integrator,
      OperatorType const &                              operator_type,
      BoundaryType const &                              boundary_type,
      dealii::types::boundary_id const                  boundary_id,
      std::shared_ptr<BoundaryDescriptor<dim> const>    boundary_descriptor,
      double const                                      time)
  {
    if(boundary_type == BoundaryType::Dirichlet)
      return coeff_times_gradient_times_normal_m;

    if(boundary_type == BoundaryType::Neumann)
    {
      if(operator_type == OperatorType::full || operator_type == OperatorType::inhomogeneous)
      {
        auto const bc       = boundary_descriptor->neumann_bc.find(boundary_id)->second;
        auto const q_points = integrator.quadrature_point(q);

        auto const h = FunctionEvaluator<value_rank, dim, Number>::value(bc, q_points, time);

        return coeff_times_gradient_times_normal_m + value_type(2.0 * h);
      }
      else if(operator_type == OperatorType::homogeneous)
        return -coeff_times_gradient_times_normal_m;
      else
        AssertThrow(false, dealii::ExcNotImplemented());
    }

    AssertThrow(false, dealii::ExcMessage("Boundary type of face is invalid or not implemented."));

    return value_type{};
  }
};
} // namespace Boundary

template<int dim>
struct OperatorData : public OperatorBaseData
{
  Operators::KernelData<dim> kernel_data{};

  std::shared_ptr<Boundary::BoundaryDescriptor<dim>> bc{};
};

template<int dim, typename Number, int n_components = 1, bool coupling_coefficient = false>
class Operator : public OperatorBase<dim, Number, n_components>
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
  initialize(dealii::MatrixFree<dim, Number> const &   matrix_free,
             dealii::AffineConstraints<Number> const & affine_constraints,
             OperatorData<dim> const &                 data);

  void
  initialize(
    dealii::MatrixFree<dim, Number> const &   matrix_free,
    dealii::AffineConstraints<Number> const & affine_constraints,
    OperatorData<dim> const &                 data_in,
    std::shared_ptr<Operators::Kernel<dim, Number, n_components, coupling_coefficient>> kernel_in);

  void
  update_coefficients();

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
  do_face_int_integral_cell_based(IntegratorFace & integrator_m,
                                  IntegratorFace & integrator_p) const override;

  void
  do_face_ext_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const override;

  void
  do_boundary_integral(IntegratorFace &                   integrator,
                       OperatorType const &               operator_type,
                       dealii::types::boundary_id const & boundary_id) const override;

  void
  do_boundary_integral_cell_based(IntegratorFace &                   integrator,
                                  OperatorType const &               operator_type,
                                  dealii::types::boundary_id const & boundary_id) const;

  OperatorData<dim> operator_data;

  std::shared_ptr<Operators::Kernel<dim, Number, n_components, coupling_coefficient>> kernel;
};
} // namespace GeneralizedLaplace
} // namespace ExaDG
#endif /* INCLUDE_EXADG_OPERATORS_GENERALIZED_LAPLACE_OPERATOR_H_ */
