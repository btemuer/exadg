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
#include "momentum_operator.h"

namespace ExaDG
{
namespace Darcy
{
template<int dim, typename Number>
void
MomentumOperator<dim, Number>::initialize(
  dealii::MatrixFree<dim, Number> const &   matrix_free,
  dealii::AffineConstraints<Number> const & affine_constraints,
  MomentumOperatorData<dim> const &         data)
{
  operator_data = data;

  Base::reinit(matrix_free, affine_constraints, data);

  // Create new objects and initialize kernels

  // Permeability operator
  {
    this->permeability_kernel = std::make_shared<Operators::PermeabilityKernel<dim, Number>>();
    this->permeability_kernel->reinit(data.permeability_kernel_data);
    this->integrator_flags =
      this->integrator_flags | this->permeability_kernel->get_integrator_flags();
  }

  // Mass operator
  {
    if(operator_data.unsteady_problem)
    {
      this->mass_kernel      = std::make_shared<MassKernel<dim, Number>>();
      this->integrator_flags = this->integrator_flags | this->mass_kernel->get_integrator_flags();
    }
  }

  // Ale momentum Operator
  {
    if(operator_data.ale)
    {
      this->ale_momentum_kernel = std::make_shared<Operators::AleMomentumKernel<dim, Number>>();
      this->ale_momentum_kernel->reinit(matrix_free,
                                        operator_data.ale_momentum_kernel_data,
                                        operator_data.dof_index,
                                        operator_data.quad_index);
      this->integrator_flags =
        this->integrator_flags | this->ale_momentum_kernel->get_integrator_flags();
    }
  }
}

template<int dim, typename Number>
MomentumOperatorData<dim> const &
MomentumOperator<dim, Number>::get_data() const
{
  return operator_data;
}

template<int dim, typename Number>
Number
MomentumOperator<dim, Number>::get_scaling_factor_mass_operator() const
{
  return this->scaling_factor_mass;
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::set_scaling_factor_mass_operator(Number const factor)
{
  this->scaling_factor_mass = factor;
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::set_grid_velocity(VectorType const & grid_velocity)
{
  ale_momentum_kernel->set_grid_velocity_ptr(grid_velocity);
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::reinit_cell(unsigned int const cell) const
{
  Base::reinit_cell(cell);

  if(operator_data.ale)
    ale_momentum_kernel->reinit_cell(cell);
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::reinit_face(unsigned int const face) const
{
  Base::reinit_face(face);

  if(operator_data.ale)
    ale_momentum_kernel->reinit_face(face);
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::reinit_boundary_face(unsigned int const face) const
{
  Base::reinit_boundary_face(face);

  if(operator_data.ale)
    ale_momentum_kernel->reinit_boundary_face(face);
}


template<int dim, typename Number>
void
MomentumOperator<dim, Number>::do_cell_integral(IntegratorCell & integrator) const
{
  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    vector const value = integrator.get_value(q);

    vector value_flux = permeability_kernel->get_volume_flux(value, integrator.quadrature_point(q));

    if(operator_data.unsteady_problem)
    {
      value_flux += mass_kernel->get_volume_flux(scaling_factor_mass, value);
    }

    if(operator_data.ale)
    {
      dyadic const gradient = integrator.get_gradient(q);
      value_flux += -ale_momentum_kernel->get_volume_flux(gradient, q);
    }

    integrator.submit_value(value_flux, q);
  }
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::do_face_integral(IntegratorFace & integrator_m,
                                                IntegratorFace & integrator_p) const
{
  if(operator_data.ale)
  {
    for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
    {
      vector const value_m  = integrator_m.get_value(q);
      vector const value_p  = integrator_p.get_value(q);
      vector const normal_m = integrator_m.get_normal_vector(q);

      vector const flux = -ale_momentum_kernel->calculate_flux(value_m, value_p, normal_m, q);

      integrator_m.submit_value(flux, q);
      integrator_p.submit_value(-flux, q);
    }
  }
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::do_face_int_integral(IntegratorFace & integrator_m,
                                                    IntegratorFace & integrator_p) const
{
  (void)integrator_p;

  if(operator_data.ale)
  {
    for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
    {
      vector const value_m = integrator_m.get_value(q);
      vector const value_p;
      vector const normal_m = integrator_m.get_normal_vector(q);

      vector const flux = -ale_momentum_kernel->calculate_flux(value_m, value_p, normal_m, q);

      integrator_m.submit_value(flux, q);
    }
  }
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::do_boundary_integral(
  IntegratorFace &                   integrator,
  const OperatorType &               operator_type,
  const dealii::types::boundary_id & boundary_id) const
{
  if(operator_data.ale)
  {
    IncNS::BoundaryTypeU const boundary_type = operator_data.bc->get_boundary_type(boundary_id);

    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      vector const value_m = IncNS::calculate_interior_value(q, integrator, operator_type);

      vector const value_p = std::invoke([&]() {
        if(operator_data.use_boundary_data == true)
          return calculate_exterior_value(value_m,
                                          q,
                                          integrator,
                                          operator_type,
                                          boundary_type,
                                          boundary_id,
                                          operator_data.bc,
                                          this->time);
        else
          return value_m;
      });

      vector const normal_m = integrator.get_normal_vector(q);
      vector const flux     = ale_momentum_kernel->calculate_flux(value_m, value_p, normal_m, q);

      integrator.submit_value(flux, q);
    }
  }
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::rhs(VectorType & rhs, double const evaluation_time) const
{
  rhs = 0;

  rhs_add(rhs, evaluation_time);
}


template<int dim, typename Number>
void
MomentumOperator<dim, Number>::rhs_add(VectorType & rhs, double const evaluation_time) const
{
  this->time = evaluation_time;

  VectorType tmp;
  tmp.reinit(rhs, false);

  this->matrix_free->loop(&This::cell_loop_inhom_operator,
                          &This::face_loop_empty,
                          &This::boundary_face_loop_inhom_operator,
                          this,
                          tmp,
                          tmp);

  // multiply by -1.0 since the results have to be shifted to the right hand side
  rhs.add(-1.0, tmp);
}


template<int dim, typename Number>
void
MomentumOperator<dim, Number>::cell_loop_inhom_operator(
  dealii::MatrixFree<dim, Number> const &       matrix_free,
  VectorType &                                  dst,
  VectorType const &                            src,
  std::pair<unsigned int, unsigned int> const & range) const
{
  if(operator_data.ale)
  {
    (void)src;
    (void)matrix_free;

    for(unsigned int cell = range.first; cell < range.second; ++cell)
    {
      this->reinit_cell(cell);

      for(unsigned int q = 0; q < this->integrator->n_q_points; ++q)
      {
        vector const grid_velocity = ale_momentum_kernel->get_grid_velocity(q);

        vector const flux =
          -permeability_kernel->get_volume_flux(grid_velocity,
                                                this->integrator->quadrature_point(q));

        this->integrator->submit_value(flux, q);
      }
      this->integrator->integrate_scatter(this->integrator_flags.cell_integrate, dst);
    }
  }
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::face_loop_empty(dealii::MatrixFree<dim, Number> const & matrix_free,
                                               VectorType &                            dst,
                                               VectorType const &                      src,
                                               Range const &                           range) const
{
  (void)matrix_free;
  (void)dst;
  (void)src;
  (void)range;

  // do nothing
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::boundary_face_loop_inhom_operator(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           range) const
{
  if(operator_data.ale)
  {
    (void)src;

    for(unsigned int face = range.first; face < range.second; face++)
    {
      this->reinit_boundary_face(face);

      // note: no gathering/evaluation is necessary when calculating the
      //       inhomogeneous part of boundary face integrals

      do_boundary_integral(*this->integrator_m,
                           OperatorType::inhomogeneous,
                           matrix_free.get_boundary_id(face));

      this->integrator_m->integrate_scatter(this->integrator_flags.face_integrate, dst);
    }
  }
}

template class MomentumOperator<2, float>;
template class MomentumOperator<2, double>;

template class MomentumOperator<3, float>;
template class MomentumOperator<3, double>;
} // namespace Darcy
} // namespace ExaDG