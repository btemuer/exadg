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

#ifndef EXADG_DARCY_GRID_VELOCITY_MANAGER_H
#define EXADG_DARCY_GRID_VELOCITY_MANAGER_H

#include <exadg/operators/lazy_ptr.h>
#include "exadg/matrix_free/integrators.h"

namespace ExaDG
{
namespace Darcy
{
template<int dim, typename Number>
class GridVelocityManager
{
private:
  using vector = dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>;

  using IntegratorCell = CellIntegrator<dim, dim, Number>;
  using IntegratorFace = FaceIntegrator<dim, dim, Number>;

  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;

public:
  void
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free,
             unsigned int const                      dof_index,
             unsigned int const                      quad_index)
  {
    integrator_grid_velocity = std::make_shared<IntegratorCell>(matrix_free, dof_index, quad_index);
    integrator_grid_velocity_face =
      std::make_shared<IntegratorFace>(matrix_free, true, dof_index, quad_index);

    matrix_free.initialize_dof_vector(grid_velocity.own(), dof_index);
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
  set_grid_velocity(VectorType const & grid_velocity_in)
  {
    grid_velocity.reset(grid_velocity_in);

    grid_velocity->update_ghost_values();
  }

  inline DEAL_II_ALWAYS_INLINE //
    vector
    get_grid_velocity_cell(unsigned int const q) const
  {
    return integrator_grid_velocity->get_value(q);
  }

  inline DEAL_II_ALWAYS_INLINE //
    vector
    get_grid_velocity_face(unsigned int const q) const
  {
    return integrator_grid_velocity_face->get_value(q);
  }

private:
  lazy_ptr<VectorType>            grid_velocity;
  std::shared_ptr<IntegratorCell> integrator_grid_velocity;
  std::shared_ptr<IntegratorFace> integrator_grid_velocity_face;
};
} // namespace Darcy
} // namespace ExaDG

#endif // EXADG_DARCY_GRID_VELOCITY_MANAGER_H
