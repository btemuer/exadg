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

#include <exadg/darcy/spatial_discretization/operators/divergence_operator.h>

namespace ExaDG
{
namespace Darcy
{
template<int dim, typename Number>
DivergenceOperator<dim, Number>::DivergenceOperator()
  : matrix_free(nullptr), time(0.0), velocity_bc(nullptr)
{
}

template<int dim, typename Number>
void
DivergenceOperator<dim, Number>::initialize(
  dealii::MatrixFree<dim, Number> const &   matrix_free_in,
  DivergenceOperatorData<dim> const & data_in)
{
  this->matrix_free = &matrix_free_in;
  this->data        = data_in;

  kernel.reinit(data_in.kernel_data);
}

template<int dim, typename Number>
void
DivergenceOperator<dim, Number>::apply(VectorType & dst, VectorType const & src) const
{
  matrix_free->loop(&This::cell_loop,
                    &This::face_loop,
                    &This::boundary_face_loop_hom_operator,
                    this,
                    dst,
                    src,
                    true /*zero_dst_vector = true*/);
}

template<int dim, typename Number>
void
DivergenceOperator<dim, Number>::apply_add(VectorType & dst, VectorType const & src) const
{
  matrix_free->loop(&This::cell_loop,
                    &This::face_loop,
                    &This::boundary_face_loop_hom_operator,
                    this,
                    dst,
                    src,
                    false /*zero_dst_vector = true*/);
}

template<int dim, typename Number>
void
DivergenceOperator<dim, Number>::rhs(VectorType & dst, Number const evaluation_time) const
{
  dst = 0.0;
  rhs_add(dst, evaluation_time);
}

template<int dim, typename Number>
void
DivergenceOperator<dim, Number>::rhs_add(VectorType & dst, Number const evaluation_time) const
{
  time = evaluation_time;

  VectorType tmp;
  tmp.reinit(dst, false /* init with 0 */);

  matrix_free->loop(&This::cell_loop_inhom_operator,
                    &This::face_loop_inhom_operator,
                    &This::boundary_face_loop_inhom_operator,
                    this,
                    tmp,
                    tmp,
                    false /*zero_dst_vector = false*/);

  // multiply by -1.0 since the boundary face integrals have to be shifted to the right hand side
  dst.add(-1.0, tmp);
}

template<int dim, typename Number>
void
DivergenceOperator<dim, Number>::do_cell_integral(CellIntegratorP & pressure,
                                                        CellIntegratorU & velocity) const
{
  for(unsigned int q = 0; q < velocity.n_q_points; ++q)
  {
    pressure.submit_gradient(kernel.get_volume_flux(velocity, q, this->time), q);
  }
}

template<int dim, typename Number>
void
DivergenceOperator<dim, Number>::do_face_integral(FaceIntegratorU & velocity_m,
                                                        FaceIntegratorU & velocity_p,
                                                        FaceIntegratorP & pressure_m,
                                                        FaceIntegratorP & pressure_p) const
{
  for(unsigned int q = 0; q < velocity_m.n_q_points; ++q)
  {
    vector value_m = velocity_m.get_value(q);
    vector value_p = velocity_p.get_value(q);

    vector const flux = kernel.calculate_flux(value_m, value_p, velocity_m.quadrature_point(q), this->time);

    scalar const flux_times_normal = flux * velocity_m.get_normal_vector(q);

    pressure_m.submit_value(flux_times_normal, q);
    // minus sign since n⁺ = - n⁻
    pressure_p.submit_value(-flux_times_normal, q);
  }
}

template<int dim, typename Number>
void
DivergenceOperator<dim, Number>::do_boundary_integral(
  FaceIntegratorU &                  velocity,
  FaceIntegratorP &                  pressure,
  OperatorType const &               operator_type,
  dealii::types::boundary_id const & boundary_id) const
{
  IncNS::BoundaryTypeU const boundary_type = data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < pressure.n_q_points; ++q)
  {
    vector const value_m = IncNS::calculate_interior_value(q, velocity, operator_type);

    vector const value_p = std::invoke([&]() {
      if(data.use_boundary_data == true)
        return calculate_exterior_value(
          value_m, q, velocity, operator_type, boundary_type, boundary_id, data.bc, time);
      else
        return value_m;
    });

    vector const flux   = kernel.calculate_flux(value_m, value_p, velocity.quadrature_point(q), this->time);
    vector const normal = velocity.get_normal_vector(q);

    pressure.submit_value(flux * normal, q);
  }
}

template<int dim, typename Number>
void
DivergenceOperator<dim, Number>::cell_loop(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           cell_range) const
{
  CellIntegratorU velocity(matrix_free, data.dof_index_velocity, data.quad_index);
  CellIntegratorP pressure(matrix_free, data.dof_index_pressure, data.quad_index);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    pressure.reinit(cell);
    velocity.reinit(cell);

    velocity.gather_evaluate(src, dealii::EvaluationFlags::values);

    do_cell_integral(pressure, velocity);

    pressure.integrate_scatter(dealii::EvaluationFlags::gradients, dst);
  }
}

template<int dim, typename Number>
void
DivergenceOperator<dim, Number>::face_loop(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           face_range) const
{
  FaceIntegratorU velocity_m(matrix_free, true, data.dof_index_velocity, data.quad_index);
  FaceIntegratorU velocity_p(matrix_free, false, data.dof_index_velocity, data.quad_index);

  FaceIntegratorP pressure_m(matrix_free, true, data.dof_index_pressure, data.quad_index);
  FaceIntegratorP pressure_p(matrix_free, false, data.dof_index_pressure, data.quad_index);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    pressure_m.reinit(face);
    pressure_p.reinit(face);

    velocity_m.reinit(face);
    velocity_p.reinit(face);

    velocity_m.gather_evaluate(src, dealii::EvaluationFlags::values);
    velocity_p.gather_evaluate(src, dealii::EvaluationFlags::values);

    do_face_integral(velocity_m, velocity_p, pressure_m, pressure_p);

    pressure_m.integrate_scatter(dealii::EvaluationFlags::values, dst);
    pressure_p.integrate_scatter(dealii::EvaluationFlags::values, dst);
  }
}

template<int dim, typename Number>
void
DivergenceOperator<dim, Number>::boundary_face_loop_hom_operator(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           face_range) const
{
  FaceIntegratorU velocity(matrix_free, true, data.dof_index_velocity, data.quad_index);

  FaceIntegratorP pressure(matrix_free, true, data.dof_index_pressure, data.quad_index);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    pressure.reinit(face);
    velocity.reinit(face);

    velocity.gather_evaluate(src, dealii::EvaluationFlags::values);

    do_boundary_integral(velocity,
                         pressure,
                         OperatorType::homogeneous,
                         matrix_free.get_boundary_id(face));

    pressure.integrate_scatter(dealii::EvaluationFlags::values, dst);
  }
}

template<int dim, typename Number>
void
DivergenceOperator<dim, Number>::cell_loop_inhom_operator(
  dealii::MatrixFree<dim, Number> const &,
  VectorType &,
  VectorType const &,
  Range const &) const
{
}

template<int dim, typename Number>
void
DivergenceOperator<dim, Number>::face_loop_inhom_operator(
  dealii::MatrixFree<dim, Number> const &,
  VectorType &,
  VectorType const &,
  Range const &) const
{
}

template<int dim, typename Number>
void
DivergenceOperator<dim, Number>::boundary_face_loop_inhom_operator(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &,
  std::pair<unsigned int, unsigned int> const & face_range) const
{
  FaceIntegratorU velocity(matrix_free, true, data.dof_index_velocity, data.quad_index);

  FaceIntegratorP pressure(matrix_free, true, data.dof_index_pressure, data.quad_index);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    pressure.reinit(face);
    velocity.reinit(face);

    do_boundary_integral(velocity,
                         pressure,
                         OperatorType::inhomogeneous,
                         matrix_free.get_boundary_id(face));

    pressure.integrate_scatter(dealii::EvaluationFlags::values, dst);
  }
}

template class DivergenceOperator<2, float>;
template class DivergenceOperator<2, double>;

template class DivergenceOperator<3, float>;
template class DivergenceOperator<3, double>;
} // namespace Darcy
} // namespace ExaDG
