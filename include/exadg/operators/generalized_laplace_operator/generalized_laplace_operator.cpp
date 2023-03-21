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

#include <exadg/operators/generalized_laplace_operator/generalized_laplace_operator.h>

namespace ExaDG
{
namespace GeneralizedLaplaceOperator
{
template<int dim, typename Number, int n_components, bool coupling_coefficient>
void
GeneralizedLaplaceOperator<dim, Number, n_components, coupling_coefficient>::initialize(
  dealii::MatrixFree<dim, Number> const &   matrix_free,
  dealii::AffineConstraints<Number> const & affine_constraints,
  GeneralizedLaplaceOperatorData<dim, Number, n_components, coupling_coefficient> const & data)
{
  Base::reinit(matrix_free, affine_constraints, data);

  operator_data = data;

  kernel->reinit(matrix_free, data.kernel_data, data.dof_index, data.quad_index);

  this->integrator_flags = kernel->get_integrator_flags();
}

template<int dim, typename Number, int n_components, bool coupling_coefficient>
void
GeneralizedLaplaceOperator<dim, Number, n_components, coupling_coefficient>::initialize(
  dealii::MatrixFree<dim, Number> const &   matrix_free,
  dealii::AffineConstraints<Number> const & affine_constraints,
  GeneralizedLaplaceOperatorData<dim, Number, n_components, coupling_coefficient> const & data,
  std::shared_ptr<
    Operators::GeneralizedLaplaceKernel<dim, Number, n_components, coupling_coefficient>>
    generalized_laplace_kernel)
{
  Base::reinit(matrix_free, affine_constraints, data);

  operator_data = data;

  kernel = generalized_laplace_kernel;

  this->integrator_flags = kernel->get_integrator_flags();
}

template<int dim, typename Number, int n_components, bool coupling_coefficient>
void
GeneralizedLaplaceOperator<dim, Number, n_components, coupling_coefficient>::update()
{
  kernel->calculate_penalty_parameter(this->get_matrix_free(), operator_data.dof_index);
  kernel->calculate_coefficients(this->get_matrix_free(), operator_data.quad_index);
}

template<int dim, typename Number, int n_components, bool coupling_coefficient>
void
GeneralizedLaplaceOperator<dim, Number, n_components, coupling_coefficient>::reinit_face(
  unsigned int const face) const
{
  Base::reinit_face(face);

  kernel->reinit_face(*this->integrator_m, *this->integrator_p, operator_data.dof_index);
}

template<int dim, typename Number, int n_components, bool coupling_coefficient>
void
GeneralizedLaplaceOperator<dim, Number, n_components, coupling_coefficient>::reinit_boundary_face(
  unsigned int const face) const
{
  Base::reinit_boundary_face(face);

  kernel->reinit_boundary_face(*this->integrator_m, operator_data.dof_index);
}

template<int dim, typename Number, int n_components, bool coupling_coefficient>
void
GeneralizedLaplaceOperator<dim, Number, n_components, coupling_coefficient>::reinit_face_cell_based(
  unsigned int const               cell,
  unsigned int const               face,
  dealii::types::boundary_id const boundary_id) const
{
  Base::reinit_face_cell_based(cell, face, boundary_id);

  kernel->reinit_face_cell_based(boundary_id,
                                 *this->integrator_m,
                                 *this->integrator_p,
                                 operator_data.dof_index);
}

template<int dim, typename Number, int n_components, bool coupling_coefficient>
void
GeneralizedLaplaceOperator<dim, Number, n_components, coupling_coefficient>::do_cell_integral(
  IntegratorCell & integrator) const
{
  for(unsigned int q = 0; q < integrator.n_q; ++q)
  {
    unsigned int const cell     = integrator.get_current_cell_index();
    Gradient const     gradient = integrator.get_gradient(q);

    Coefficient const coefficient = kernel->get_coefficient_cell(cell, q);
    Gradient const    volume_flux = kernel->get_volume_flux(gradient, coefficient);

    integrator.submit_gradient(volume_flux, q);
  }
}

template<int dim, typename Number, int n_components, bool coupling_coefficient>
void
GeneralizedLaplaceOperator<dim, Number, n_components, coupling_coefficient>::do_face_integral(
  IntegratorFace & integrator_m,
  IntegratorFace & integrator_p) const
{
  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    unsigned int const cell = integrator_m.get_current_cell_index();

    Value const value_m = integrator_m.get_value(q);
    Value const value_p = integrator_p.get_value(q);

    Gradient const gradient_m = integrator_m.get_gradient(q);
    Gradient const gradient_p = integrator_p.get_gradient(q);

    vector const normal_m = integrator_m.get_normal_vector(q);

    Coefficient const coefficient = kernel->get_coefficient(cell, q);

    Gradient const gradient_flux =
      kernel->get_gradient_flux(value_m, value_p, normal_m, coefficient);

    Value const value_flux =
      kernel->get_value_flux(gradient_m, gradient_p, value_m, value_p, normal_m, coefficient);

    integrator_m.submit_gradient(gradient_flux, q);
    integrator_p.submit_gradient(gradient_flux, q);

    integrator_m.submit_value(value_flux, q);
    integrator_p.submit_value(-value_flux, q); // - sign since n⁺ = -n⁻
  }
}

template<int dim, typename Number, int n_components, bool coupling_coefficient>
void
GeneralizedLaplaceOperator<dim, Number, n_components, coupling_coefficient>::do_face_int_integral(
  IntegratorFace & integrator_m,
  IntegratorFace & integrator_p,
  bool const       revert_int_ext) const
{
  (void)integrator_p;

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    unsigned int const face = integrator_m.get_current_cell_index();

    Value const value_m = integrator_m.get_value(q);
    Value const value_p; // set exterior values to zero

    Gradient const gradient_m = integrator_m.get_gradient(q);
    Gradient const gradient_p; // set exterior gradients to zero

    vector const normal_m =
      (revert_int_ext) ? -integrator_m.get_normal_vector(q) : integrator_m.get_normal_vector(q);

    Coefficient const coefficient = kernel->get_coefficient_face(face, q);

    Gradient const gradient_flux =
      kernel->get_gradient_flux(value_m, value_p, normal_m, coefficient);

    Value const value_flux =
      kernel->get_value_flux(gradient_m, gradient_p, value_m, value_p, normal_m, coefficient);

    integrator_m.submit_gradient(gradient_flux, q);
    integrator_m.submit_value(value_flux, q);
  }
}

template<int dim, typename Number, int n_components, bool coupling_coefficient>
void
GeneralizedLaplaceOperator<dim, Number, n_components, coupling_coefficient>::do_face_ext_integral(
  IntegratorFace & integrator_m,
  IntegratorFace & integrator_p) const
{
  // call do_face_int_integral() with the reverse order and set revert_int_ext to true
  do_face_int_integral(integrator_p, integrator_m, true);
}

template<int dim, typename Number, int n_components, bool coupling_coefficient>
void
GeneralizedLaplaceOperator<dim, Number, n_components, coupling_coefficient>::do_boundary_integral(
  IntegratorFace &                   integrator,
  const OperatorType &               operator_type,
  const dealii::types::boundary_id & boundary_id) const
{
  auto const boundary_type = operator_data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    unsigned int const face = integrator.get_current_cell_index();

    Value const value_m = BC::calculate_interior_value(q, integrator, operator_type);

    Value const value_p = BC::calculate_exterior_value(value_m,
                                                       q,
                                                       integrator,
                                                       operator_type,
                                                       boundary_type,
                                                       boundary_id,
                                                       operator_data.bc,
                                                       this->time);

    vector const normal_m = integrator.get_normal_vector(q);

    Coefficient const coefficient = kernel->get_coefficient_face(face, q);

    Gradient const gradient_flux =
      kernel->get_gradient_flux(value_m, value_p, normal_m, coefficient);

    Value const coeff_times_normal_gradient_m =
      BC::calculate_interior_coeff_times_normal_gradient(q, integrator, boundary_type, coefficient);

    Value const coeff_times_normal_gradient_p =
      BC::calculate_exterior_coeff_times_normal_gradient(coeff_times_normal_gradient_m,
                                                         q,
                                                         integrator,
                                                         operator_type,
                                                         boundary_type,
                                                         boundary_id,
                                                         operator_data.bc,
                                                         this->time);

    Value const value_flux = kernel->get_value_flux(coeff_times_normal_gradient_m,
                                                    coeff_times_normal_gradient_p,
                                                    value_m,
                                                    value_p,
                                                    normal_m,
                                                    coefficient);

    integrator.submit_gradient(gradient_flux, q);
    integrator.submit_value(value_flux, q);
  }
}
} // namespace GeneralizedLaplaceOperator
} // namespace ExaDG
