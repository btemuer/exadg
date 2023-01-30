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

#ifndef EXADG_DARCY_ALE_MOMENTUM_OPERATOR_H
#define EXADG_DARCY_ALE_MOMENTUM_OPERATOR_H

#include <exadg/darcy/user_interface/parameters.h>
#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/operator_base.h>

namespace ExaDG
{
namespace Darcy
{
namespace Operators
{
struct AleMomentumKernelData
{
  double upwind_factor{1.0};
};

template<int dim, typename Number>
class AleMomentumKernel
{
private:
  using scalar = dealii::VectorizedArray<Number>;
  using vector = dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>;
  using dyadic = dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>;

  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;

  using IntegratorCell = CellIntegrator<dim, dim, Number>;
  using IntegratorFace = FaceIntegrator<dim, dim, Number>;

public:
  void
  reinit(dealii::MatrixFree<dim, Number> const & matrix_free,
         AleMomentumKernelData const &           data_in,
         unsigned int const                      dof_index,
         unsigned int const                      quad_index)
  {
    data = data_in;

    integrator_grid_velocity = std::make_shared<IntegratorCell>(matrix_free, dof_index, quad_index);
    integrator_grid_velocity_face =
      std::make_shared<IntegratorFace>(matrix_free, true, dof_index, quad_index);

    matrix_free.initialize_dof_vector(grid_velocity.own(), dof_index);
  }

  static MappingFlags
  get_mapping_flags()
  {
    MappingFlags flags;

    flags.cells          = dealii::update_JxW_values | dealii::update_gradients;
    flags.inner_faces    = dealii::update_JxW_values | dealii::update_normal_vectors;
    flags.boundary_faces = dealii::update_JxW_values | dealii::update_normal_vectors;

    return flags;
  }

  IntegratorFlags
  get_integrator_flags() const
  {
    IntegratorFlags flags;

    flags.cell_evaluate  = dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients;
    flags.cell_integrate = dealii::EvaluationFlags::values;

    flags.face_evaluate  = dealii::EvaluationFlags::values;
    flags.face_integrate = dealii::EvaluationFlags::values;

    return flags;
  }

  void
  reinit_cell(unsigned int const cell) const
  {
    integrator_grid_velocity->reinit(cell);
    integrator_grid_velocity->gather_evaluate(*grid_velocity, dealii::EvaluationFlags::values);
  }

  void
  reinit_face(unsigned int const face) const
  {
    integrator_grid_velocity_face->reinit(face);
    integrator_grid_velocity_face->gather_evaluate(*grid_velocity, dealii::EvaluationFlags::values);
  }

  void
  reinit_boundary_face(unsigned int const face) const
  {
    integrator_grid_velocity_face->reinit(face);
    integrator_grid_velocity_face->gather_evaluate(*grid_velocity, dealii::EvaluationFlags::values);
  }

  void
  set_grid_velocity_ptr(VectorType const & grid_velocity_in)
  {
    grid_velocity.reset(grid_velocity_in);

    grid_velocity->update_ghost_values();
  }

  VectorType const &
  get_grid_velocity_vector() const
  {
    return *grid_velocity;
  }

  inline DEAL_II_ALWAYS_INLINE //
    vector
    get_volume_flux(dyadic const & velocity_gradient, unsigned int const q)
  {
    vector const grid_velocity_value = integrator_grid_velocity->get_value(q);

    return velocity_gradient * grid_velocity_value;
  }

  inline DEAL_II_ALWAYS_INLINE //
    vector
    get_grid_velocity(unsigned int const q)
  {
    return integrator_grid_velocity->get_value(q);
  }

  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_flux(vector const &     velocity_value_m,
                   vector const &     velocity_value_P,
                   vector const &     normal_m,
                   unsigned int const q)
  {
    vector grid_velocity_value  = integrator_grid_velocity_face->get_value(q);
    scalar normal_grid_velocity = grid_velocity_value * normal_m;

    //vector average_velocity = 0.5 * (velocity_value_m + velocity_value_P);
    vector velocity_jump    = velocity_value_m - velocity_value_P;

    vector flux = 0.5 * velocity_jump *
                  (data.upwind_factor * std::abs(normal_grid_velocity) - normal_grid_velocity);

    return flux;
  }

private:
  AleMomentumKernelData data;

  lazy_ptr<VectorType> grid_velocity;

  std::shared_ptr<IntegratorCell> integrator_grid_velocity;
  std::shared_ptr<IntegratorFace> integrator_grid_velocity_face;
};

} // namespace Operators
} // namespace Darcy
} // namespace ExaDG

#endif // EXADG_DARCY_ALE_MOMENTUM_OPERATOR_H
