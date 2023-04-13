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
namespace GeneralizedLaplace
{
template<int dim, typename Number, int n_components, bool coupling_coefficient>
void
Operator<dim, Number, n_components, coupling_coefficient>::initialize(
  dealii::MatrixFree<dim, Number> const &   matrix_free,
  dealii::AffineConstraints<Number> const & affine_constraints,
  OperatorData<dim> const &                 data)
{
  Base::reinit(matrix_free, affine_constraints, data);

  operator_data = data;

  kernel->reinit(matrix_free, data.kernel_data, data.dof_index, data.quad_index);

  this->integrator_flags = kernel->get_integrator_flags();
}

template<int dim, typename Number, int n_components, bool coupling_coefficient>
void
Operator<dim, Number, n_components, coupling_coefficient>::initialize(
  dealii::MatrixFree<dim, Number> const &   matrix_free,
  dealii::AffineConstraints<Number> const & affine_constraints,
  OperatorData<dim> const &                 data_in,
  std::shared_ptr<Operators::Kernel<dim, Number, n_components, coupling_coefficient>> const
    kernel_in)
{
  Base::reinit(matrix_free, affine_constraints, data_in);

  operator_data = data_in;

  kernel = kernel_in;

  this->integrator_flags = kernel->get_integrator_flags();
}

template<int dim, typename Number, int n_components, bool coupling_coefficient>
void
Operator<dim, Number, n_components, coupling_coefficient>::update_coefficients()
{
  kernel->update_coefficients(this->get_matrix_free(), operator_data.quad_index, this->time);
}

template<int dim, typename Number, int n_components, bool coupling_coefficient>
void
Operator<dim, Number, n_components, coupling_coefficient>::reinit_face(
  unsigned int const face) const
{
  Base::reinit_face(face);

  kernel->reinit_face(*this->integrator_m, *this->integrator_p, operator_data.dof_index);
}

template<int dim, typename Number, int n_components, bool coupling_coefficient>
void
Operator<dim, Number, n_components, coupling_coefficient>::reinit_boundary_face(
  unsigned int const face) const
{
  Base::reinit_boundary_face(face);

  kernel->reinit_boundary_face(*this->integrator_m, operator_data.dof_index);
}

template<int dim, typename Number, int n_components, bool coupling_coefficient>
void
Operator<dim, Number, n_components, coupling_coefficient>::reinit_face_cell_based(
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
Operator<dim, Number, n_components, coupling_coefficient>::do_cell_integral(
  IntegratorCell & integrator) const
{
  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    unsigned int const  cell     = integrator.get_current_cell_index();
    gradient_type const gradient = integrator.get_gradient(q);

    coefficient_type const coefficient = kernel->get_coefficient_cell(cell, q);
    gradient_type const    volume_flux = kernel->get_volume_flux(gradient, coefficient);

    integrator.submit_gradient(volume_flux, q);
  }
}

template<int dim, typename Number, int n_components, bool coupling_coefficient>
void
Operator<dim, Number, n_components, coupling_coefficient>::do_face_integral(
  IntegratorFace & integrator_m,
  IntegratorFace & integrator_p) const
{
  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    unsigned int const face = integrator_m.get_current_cell_index();

    value_type const value_m = integrator_m.get_value(q);
    value_type const value_p = integrator_p.get_value(q);

    gradient_type const gradient_m = integrator_m.get_gradient(q);
    gradient_type const gradient_p = integrator_p.get_gradient(q);

    vector const normal_m = integrator_m.get_normal_vector(q);

    coefficient_type const coefficient = kernel->get_coefficient_face(face, q);

    gradient_type const gradient_flux =
      kernel->calculate_gradient_flux(value_m, value_p, normal_m, coefficient);

    value_type const value_flux =
      kernel->calculate_value_flux(gradient_m, gradient_p, value_m, value_p, normal_m, coefficient);

    integrator_m.submit_gradient(gradient_flux, q);
    integrator_p.submit_gradient(gradient_flux, q);

    integrator_m.submit_value(value_flux, q);
    integrator_p.submit_value(-value_flux, q); // - sign since n⁺ = -n⁻
  }
}

template<int dim, typename Number, int n_components, bool coupling_coefficient>
void
Operator<dim, Number, n_components, coupling_coefficient>::do_face_int_integral(
  IntegratorFace & integrator_m,
  IntegratorFace & integrator_p) const
{
  (void)integrator_p;

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    unsigned int const face = integrator_m.get_current_cell_index();

    value_type const value_m = integrator_m.get_value(q);
    value_type const value_p; // set exterior values to zero

    gradient_type const gradient_m = integrator_m.get_gradient(q);
    gradient_type const gradient_p; // set exterior gradients to zero

    vector const normal_m = integrator_m.get_normal_vector(q);

    coefficient_type const coefficient = kernel->get_coefficient_face(face, q);

    gradient_type const gradient_flux =
      kernel->calculate_gradient_flux(value_m, value_p, normal_m, coefficient);

    value_type const value_flux =
      kernel->calculate_value_flux(gradient_m, gradient_p, value_m, value_p, normal_m, coefficient);

    integrator_m.submit_gradient(gradient_flux, q);
    integrator_m.submit_value(value_flux, q);
  }
}

template<int dim, typename Number, int n_components, bool coupling_coefficient>
void
Operator<dim, Number, n_components, coupling_coefficient>::do_face_ext_integral(
  IntegratorFace & integrator_m,
  IntegratorFace & integrator_p) const
{
  (void)integrator_m;

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    unsigned int const face = integrator_m.get_current_cell_index();

    value_type const value_m; // set interior values to zero
    value_type const value_p = integrator_p.get_value(q);

    gradient_type const gradient_m; // set interior gradients to zero
    gradient_type const gradient_p = integrator_p.get_gradient(q);

    // multiply by -1.0 to get the correct normal vector
    vector const normal_p = -integrator_p.get_normal_vector(q);

    coefficient_type const coefficient = kernel->get_coefficient_face(face, q);

    gradient_type const gradient_flux =
      kernel->calculate_gradient_flux(value_p, value_m, normal_p, coefficient);

    value_type const value_flux =
      kernel->calculate_value_flux(gradient_p, gradient_m, value_p, value_m, normal_p, coefficient);

    integrator_p.submit_gradient(gradient_flux, q);
    integrator_p.submit_value(value_flux, q);
  }
}

template<int dim, typename Number, int n_components, bool coupling_coefficient>
void
Operator<dim, Number, n_components, coupling_coefficient>::do_boundary_integral(
  IntegratorFace &                   integrator,
  const OperatorType &               operator_type,
  const dealii::types::boundary_id & boundary_id) const
{
  auto const boundary_type = operator_data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    unsigned int const face = integrator.get_current_cell_index();

    value_type const value_m = BC::calculate_interior_value(q, integrator, operator_type);

    value_type const value_p = BC::calculate_exterior_value(value_m,
                                                            q,
                                                            integrator,
                                                            operator_type,
                                                            boundary_type,
                                                            boundary_id,
                                                            operator_data.bc,
                                                            this->time);

    vector const normal_m = integrator.get_normal_vector(q);

    coefficient_type const coefficient = kernel->get_coefficient_face(face, q);

    gradient_type const gradient_flux =
      kernel->calculate_gradient_flux(value_m, value_p, normal_m, coefficient);

    value_type const coeff_times_normal_gradient_m =
      BC::calculate_interior_coeff_times_normal_gradient(q, integrator, operator_type, coefficient);

    value_type const coeff_times_normal_gradient_p =
      BC::calculate_exterior_coeff_times_normal_gradient(coeff_times_normal_gradient_m,
                                                         q,
                                                         integrator,
                                                         operator_type,
                                                         boundary_type,
                                                         boundary_id,
                                                         operator_data.bc,
                                                         this->time);

    value_type const value_flux = kernel->calculate_value_flux(coeff_times_normal_gradient_m,
                                                               coeff_times_normal_gradient_p,
                                                               value_m,
                                                               value_p,
                                                               normal_m,
                                                               coefficient);

    integrator.submit_gradient(gradient_flux, q);
    integrator.submit_value(value_flux, q);
  }
}

template class Operator<2, float, 1, false>;
template class Operator<3, float, 1, false>;

template class Operator<2, float, 2, false>;
template class Operator<2, float, 2, true>;

template class Operator<3, float, 3, false>;
template class Operator<3, float, 3, true>;

template class Operator<2, double, 1, false>;
template class Operator<3, double, 1, false>;

template class Operator<2, double, 2, false>;
template class Operator<2, double, 2, true>;

template class Operator<3, double, 3, false>;
template class Operator<3, double, 3, true>;

} // namespace GeneralizedLaplace
} // namespace ExaDG
