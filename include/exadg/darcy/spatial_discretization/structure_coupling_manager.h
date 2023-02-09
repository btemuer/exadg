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

#include <exadg/functions_and_boundary_conditions/evaluate_functions.h>
#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/lazy_ptr.h>

namespace ExaDG
{
namespace Darcy
{
template<int dim, typename Number>
class StructureCouplingManager
{
private:
  using point = dealii::Point<dim, dealii::VectorizedArray<Number>>;

  using scalar = dealii::VectorizedArray<Number>;
  using vector = dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>;
  using dyadic = dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>;

  using IntegratorCell = CellIntegrator<dim, dim, Number>;
  using IntegratorFace = FaceIntegrator<dim, dim, Number>;

  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;

public:
  void
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free,
             unsigned int const                    dof_index,
             unsigned int const                    quad_index)
  {
    grid_displacement_cell_integrator =
      std::make_shared<IntegratorCell>(matrix_free, dof_index, quad_index);
    grid_displacement_face_integrator =
      std::make_shared<IntegratorFace>(matrix_free, true, dof_index, quad_index);

    grid_velocity_cell_integrator =
      std::make_shared<IntegratorCell>(matrix_free, dof_index, quad_index);
    grid_velocity_face_integrator =
      std::make_shared<IntegratorFace>(matrix_free, true, dof_index, quad_index);

    matrix_free.initialize_dof_vector(grid_displacement_dof_vector.own(), dof_index);
    matrix_free.initialize_dof_vector(grid_velocity_dof_vector.own(), dof_index);
  }

  void
  reinit_gather_evaluate_displacement_cell(unsigned int const cell) const
  {
    grid_displacement_cell_integrator->reinit(cell);

    grid_displacement_cell_integrator->gather_evaluate(*grid_displacement_dof_vector,
                                                       dealii::EvaluationFlags::gradients);
  }

  void
  reinit_gather_evaluate_displacement_face(unsigned int const face) const
  {
    grid_displacement_face_integrator->reinit(face);

    grid_displacement_face_integrator->gather_evaluate(*grid_displacement_dof_vector,
                                                       dealii::EvaluationFlags::gradients);
  }

  void
  reinit_gather_evaluate_velocity_cell(unsigned int const cell) const
  {
    grid_velocity_cell_integrator->reinit(cell);

    grid_velocity_cell_integrator->gather_evaluate(*grid_velocity_dof_vector,
                                                   dealii::EvaluationFlags::values);
  }

  void
  reinit_gather_evaluate_velocity_face(unsigned int const face) const
  {
    grid_velocity_face_integrator->reinit(face);

    grid_velocity_face_integrator->gather_evaluate(*grid_velocity_dof_vector,
                                                   dealii::EvaluationFlags::values);
  }


  void
  set_grid_velocity(VectorType const & grid_displacement_dof_vector_in,
                    VectorType const & grid_velocity_dof_vector_in)
  {
    grid_displacement_dof_vector.reset(grid_displacement_dof_vector_in);
    grid_velocity_dof_vector.reset(grid_velocity_dof_vector_in);

    grid_displacement_dof_vector->update_ghost_values();
    grid_velocity_dof_vector->update_ghost_values();
  }

  inline DEAL_II_ALWAYS_INLINE //
    vector
    get_grid_velocity(bool const is_face, unsigned int const q) const
  {
    return is_face ? grid_velocity_face_integrator->get_value(q) :
                     grid_velocity_cell_integrator->get_value(q);
  }

  inline DEAL_II_ALWAYS_INLINE //
    scalar
    compute_porosity(bool const                             is_face,
                     std::shared_ptr<dealii::Function<dim>> initial_porosity_field,
                     unsigned int const                     q) const
  {
    scalar const initial_porosity =
      is_face ?
        FunctionEvaluator<0, dim, Number>::value(
          initial_porosity_field, grid_displacement_face_integrator->quadrature_point(q), 0.0) :
        FunctionEvaluator<0, dim, Number>::value(
          initial_porosity_field, grid_displacement_cell_integrator->quadrature_point(q), 0.0);

    return 1.0 + (initial_porosity - 1.0) / get_detF(is_face, q);
  }

private:
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    get_detF(bool const is_face, unsigned int const q) const
  {
    dyadic const deformation_gradient = is_face ?
                                          grid_displacement_face_integrator->get_gradient(q) :
                                          grid_displacement_cell_integrator->get_gradient(q);

    return determinant(deformation_gradient + dealii::unit_symmetric_tensor<dim>());
  }

private:
  lazy_ptr<VectorType> grid_displacement_dof_vector;
  lazy_ptr<VectorType> grid_velocity_dof_vector;

  std::shared_ptr<IntegratorCell> grid_displacement_cell_integrator;
  std::shared_ptr<IntegratorFace> grid_displacement_face_integrator;

  std::shared_ptr<IntegratorCell> grid_velocity_cell_integrator;
  std::shared_ptr<IntegratorFace> grid_velocity_face_integrator;
};
} // namespace Darcy
} // namespace ExaDG

#endif // EXADG_DARCY_GRID_VELOCITY_MANAGER_H
