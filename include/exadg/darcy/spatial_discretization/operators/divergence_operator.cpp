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
  dealii::MatrixFree<dim, Number> const &                matrix_free_in,
  DivergenceOperatorData<dim> const &                    data_in,
  std::shared_ptr<StructureCouplingManager<dim, Number>> structure_coupling_manager_in)
{
  this->matrix_free                = &matrix_free_in;
  this->data                       = data_in;
  this->structure_coupling_manager = structure_coupling_manager_in;

  kernel.reinit(data_in.kernel_data);
}

template<int dim, typename Number>
void
DivergenceOperator<dim, Number>::apply(VectorType & dst, VectorType const & src) const
{
  dst = 0.0;
  apply_add(dst, src);
}

template<int dim, typename Number>
void
DivergenceOperator<dim, Number>::apply_add(VectorType & dst, VectorType const & src) const
{
  matrix_free->loop(&This::cell_loop_hom_operator,
                    &This::face_loop_hom_operator,
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
DivergenceOperator<dim, Number>::cell_loop_hom_operator(
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

    if (data.ale)
      structure_coupling_manager->reinit_gather_evaluate_displacement_cell(cell);

    for(unsigned int q = 0; q < velocity.n_q_points; ++q)
    {
      scalar const porosity = calculate_porosity(false, velocity, q);

      pressure.submit_gradient(kernel.get_volume_flux(velocity.get_value(q), porosity), q);
    }

    pressure.integrate_scatter(dealii::EvaluationFlags::gradients, dst);
  }
}


template<int dim, typename Number>
void
DivergenceOperator<dim, Number>::cell_loop_inhom_operator(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           range) const
{
  (void)src;

  if(data.ale)
  {
    CellIntegratorP pressure(matrix_free, data.dof_index_pressure, data.quad_index);

    for(unsigned int cell = range.first; cell < range.second; ++cell)
    {
      pressure.reinit(cell);

      structure_coupling_manager->reinit_gather_evaluate_displacement_cell(cell);
      structure_coupling_manager->reinit_gather_evaluate_velocity_cell(cell);

      for(unsigned int q = 0; q < pressure.n_q_points; ++q)
      {
        scalar const porosity = calculate_porosity(false, pressure, q);

        pressure.submit_gradient(
          kernel.get_volume_flux(structure_coupling_manager->get_grid_velocity(false, q),
                                 1.0 - porosity),
          q);
      }
      pressure.integrate_scatter(dealii::EvaluationFlags::gradients, dst);
    }
  }
}

template<int dim, typename Number>
void
DivergenceOperator<dim, Number>::face_loop_hom_operator(
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

    if (data.ale)
      structure_coupling_manager->reinit_gather_evaluate_displacement_face(face);

    for(unsigned int q = 0; q < velocity_m.n_q_points; ++q)
    {
      vector value_m = velocity_m.get_value(q);
      vector value_p = velocity_p.get_value(q);

      scalar const porosity = calculate_porosity(true, velocity_m, q);

      vector const flux = kernel.calculate_flux(value_m, value_p, porosity);

      scalar const flux_times_normal = flux * velocity_m.get_normal_vector(q);

      pressure_m.submit_value(flux_times_normal, q);
      // minus sign since n⁺ = - n⁻
      pressure_p.submit_value(-flux_times_normal, q);
    }

    pressure_m.integrate_scatter(dealii::EvaluationFlags::values, dst);
    pressure_p.integrate_scatter(dealii::EvaluationFlags::values, dst);
  }
}


template<int dim, typename Number>
void
DivergenceOperator<dim, Number>::face_loop_inhom_operator(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           face_range) const
{
  (void)src;

  if(data.ale)
  {
    FaceIntegratorP pressure_m(matrix_free, true, data.dof_index_pressure, data.quad_index);
    FaceIntegratorP pressure_p(matrix_free, false, data.dof_index_pressure, data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      pressure_m.reinit(face);
      pressure_p.reinit(face);

      structure_coupling_manager->reinit_gather_evaluate_displacement_face(face);
      structure_coupling_manager->reinit_gather_evaluate_velocity_face(face);

      for(unsigned int q = 0; q < pressure_m.n_q_points; ++q)
      {
        vector const grid_velocity_value = structure_coupling_manager->get_grid_velocity(true, q);

        scalar const porosity = calculate_porosity(true, pressure_m, q);

        vector const flux = (1.0 - porosity) * grid_velocity_value;

        scalar const flux_times_normal = flux * pressure_m.get_normal_vector(q);

        pressure_m.submit_value(flux_times_normal, q);
        // minus sign since n⁺ = - n⁻
        pressure_p.submit_value(-flux_times_normal, q);
      }

      pressure_m.integrate_scatter(dealii::EvaluationFlags::values, dst);
      pressure_p.integrate_scatter(dealii::EvaluationFlags::values, dst);
    }
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

    if (data.ale)
      structure_coupling_manager->reinit_gather_evaluate_displacement_face(face);

    velocity.gather_evaluate(src, dealii::EvaluationFlags::values);

    IncNS::BoundaryTypeU const boundary_type =
      data.bc->get_boundary_type(matrix_free.get_boundary_id(face));

    for(unsigned int q = 0; q < pressure.n_q_points; ++q)
    {
      vector const value_m =
        IncNS::calculate_interior_value(q, velocity, OperatorType::homogeneous);

      vector const value_p = std::invoke([&]() {
        if(data.use_boundary_data == true)
          return IncNS::calculate_exterior_value(value_m,
                                                 q,
                                                 velocity,
                                                 OperatorType::homogeneous,
                                                 boundary_type,
                                                 matrix_free.get_boundary_id(face),
                                                 data.bc,
                                                 this->time);
        else
          return value_m;
      });

      scalar const porosity = calculate_porosity(true, pressure, q);

      vector const flux   = kernel.calculate_flux(value_m, value_p, porosity);
      vector const normal = velocity.get_normal_vector(q);

      pressure.submit_value(flux * normal, q);
    }

    pressure.integrate_scatter(dealii::EvaluationFlags::values, dst);
  }
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

    if(data.ale)
      structure_coupling_manager->reinit_gather_evaluate_displacement_face(face);

    IncNS::BoundaryTypeU const boundary_type =
      data.bc->get_boundary_type(matrix_free.get_boundary_id(face));

    for(unsigned int q = 0; q < velocity.n_q_points; ++q)
    {
      vector const value_m;

      scalar const porosity = calculate_porosity(true, velocity, q);

      vector const value_p = std::invoke([&]() {
        if(data.use_boundary_data == true)
          return calculate_exterior_value(value_m,
                                          q,
                                          velocity,
                                          OperatorType::inhomogeneous,
                                          boundary_type,
                                          matrix_free.get_boundary_id(face),
                                          data.bc,
                                          this->time);
        else
          return value_m;
      });

      vector       flux   = kernel.calculate_flux(value_m, value_p, porosity);
      vector const normal = velocity.get_normal_vector(q);

      if(data.ale)
      {
        structure_coupling_manager->reinit_gather_evaluate_velocity_face(face);
        vector const grid_velocity_value = structure_coupling_manager->get_grid_velocity(true, q);
        flux += (1.0 - porosity) * grid_velocity_value;
      }

      pressure.submit_value(flux * normal, q);
    }

    pressure.integrate_scatter(dealii::EvaluationFlags::values, dst);
  }
}

template<int dim, typename Number>
template<typename Integrator>
typename DivergenceOperator<dim, Number>::scalar
DivergenceOperator<dim, Number>::calculate_porosity(bool const         is_face,
                                                    Integrator const & integrator,
                                                    unsigned int const q) const
{
  if(data.ale)
    return structure_coupling_manager->compute_porosity(is_face,
                                                        data.kernel_data.initial_porosity_field,
                                                        q);
  else
    return FunctionEvaluator<0, dim, Number>::value(data.kernel_data.initial_porosity_field,
                                                    integrator.quadrature_point(q),
                                                    0.0);
}

template class DivergenceOperator<2, float>;
template class DivergenceOperator<2, double>;

template class DivergenceOperator<3, float>;
template class DivergenceOperator<3, double>;
} // namespace Darcy
} // namespace ExaDG
