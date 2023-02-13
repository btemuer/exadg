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
#include <exadg/grid/grid_motion_interface.h>
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
  StructureCouplingManager(dealii::MatrixFree<dim, Number> const & matrix_free,
                           unsigned int const                      dof_index,
                           unsigned int const                      quad_index,
                           dealii::Mapping<dim> const &            reference_mapping,
                           std::shared_ptr<dealii::Function<dim>>  initial_porosity_field)
    : // grid_coordinates_cell_integrator(reference_mapping,
      //                                 matrix_free.get_dof_handler(dof_index).get_fe(),
      //                                 matrix_free.get_quadrature(quad_index).get_tensor_basis()[0],
      //                                 dealii::update_quadrature_points | dealii::update_gradients),
      grid_coordinates_cell_integrator(reference_mapping,
                                       matrix_free.get_dof_handler(dof_index).get_fe(),
                                       matrix_free.get_quadrature(quad_index),
                                       dealii::update_quadrature_points | dealii::update_gradients),
      grid_coordinates_face_integrator(reference_mapping,
                                       matrix_free.get_dof_handler(dof_index).get_fe(),
                                       matrix_free.get_face_quadrature(quad_index),
                                       dealii::update_quadrature_points | dealii::update_gradients),
      grid_velocity_cell_integrator(matrix_free, dof_index, quad_index),
      grid_velocity_face_integrator(matrix_free, true, dof_index, quad_index),
      initial_porosity_field(initial_porosity_field)
  {
    matrix_free.initialize_dof_vector(grid_coordinates_dof_vector.own(), dof_index);
    matrix_free.initialize_dof_vector(grid_velocity_dof_vector.own(), dof_index);

    // porosity_cell.resize(grid_coordinates_cell_integrator.n_q_points);
    porosity_cell.resize(grid_coordinates_cell_integrator.get_quadrature().size());
    porosity_face.resize(grid_coordinates_face_integrator.get_quadrature().size());
  }

  void
  reinit_gather_evaluate_velocity_cell(unsigned int const cell)
  {
    grid_velocity_cell_integrator.reinit(cell);

    grid_velocity_cell_integrator.gather_evaluate(*grid_velocity_dof_vector,
                                                  dealii::EvaluationFlags::values);
  }

  void
  reinit_gather_evaluate_velocity_face(unsigned int const face)
  {
    grid_velocity_face_integrator.reinit(face);

    grid_velocity_face_integrator.gather_evaluate(*grid_velocity_dof_vector,
                                                  dealii::EvaluationFlags::values);
  }

  /*
  void
  reinit_gather_evaluate_porosity_cell(
    std::vector<typename dealii::DoFHandler<dim>::cell_iterator> const & cell_iterators)
  {
    for(unsigned int v = 0; v < cell_iterators.size(); ++v)
    {
      grid_coordinates_cell_integrator.reinit(cell_iterators[v]);

      grid_coordinates_cell_integrator.gather_evaluate(*grid_coordinates_dof_vector,
                                                       dealii::EvaluationFlags::gradients);

      for(unsigned int q = 0; q < porosity_face.size(); ++q)
      {
        dealii::Point<dim> const single_q_point =
          batch_to_single_q_point(grid_coordinates_cell_integrator.quadrature_point(q), v);

        dealii::Tensor<2, dim, Number> const single_gradient =
          batch_to_single_dyadic(grid_coordinates_cell_integrator.get_gradient(q), v);

        porosity_cell[q][v] = compute_porosity(single_q_point, single_gradient);
      }
    }
  }
*/
  void
  reinit_gather_evaluate_porosity(
    std::vector<typename dealii::DoFHandler<dim>::cell_iterator> const & cell_iterators)
  {
    for(unsigned int v = 0; v < cell_iterators.size(); ++v)
    {
      grid_coordinates_cell_integrator.reinit(cell_iterators[v]);

      std::vector<std::vector<dealii::Tensor<1, dim, Number>>> gradients(
        porosity_cell.size(), std::vector<dealii::Tensor<1, dim, Number>>(dim));
      grid_coordinates_cell_integrator.get_function_gradients(*grid_coordinates_dof_vector,
                                                              gradients);

      for(unsigned int q = 0; q < porosity_cell.size(); ++q)
        porosity_cell[q][v] = compute_porosity(grid_coordinates_cell_integrator.quadrature_point(q),
                                               to_dyadic(gradients[q]));
    }
  }

  void
  reinit_gather_evaluate_porosity(
    std::vector<std::pair<typename dealii::DoFHandler<dim>::cell_iterator, unsigned int>> const &
      face_iterators)
  {
    for(unsigned int v = 0; v < face_iterators.size(); ++v)
    {
      grid_coordinates_face_integrator.reinit(face_iterators[v].first, face_iterators[v].second);

      std::vector<std::vector<dealii::Tensor<1, dim, Number>>> gradients(
        porosity_face.size(), std::vector<dealii::Tensor<1, dim, Number>>(dim));
      grid_coordinates_face_integrator.get_function_gradients(*grid_coordinates_dof_vector,
                                                              gradients);

      for(unsigned int q = 0; q < porosity_face.size(); ++q)
        porosity_face[q][v] = compute_porosity(grid_coordinates_face_integrator.quadrature_point(q),
                                               to_dyadic(gradients[q]));
    }
  }

  scalar
  get_porosity(bool const is_face, unsigned int const q)
  {
    return is_face ? porosity_face[q] : porosity_cell[q];
  }

  void
  set_grid_coordinates_and_velocity(VectorType const & grid_coordinates_dof_vector_in,
                                    VectorType const & grid_velocity_dof_vector_in)
  {
    grid_coordinates_dof_vector.reset(grid_coordinates_dof_vector_in);
    grid_coordinates_dof_vector->update_ghost_values();

    grid_velocity_dof_vector.reset(grid_velocity_dof_vector_in);
    grid_velocity_dof_vector->update_ghost_values();
  }

  inline DEAL_II_ALWAYS_INLINE //
    vector
    get_grid_velocity(bool const is_face, unsigned int const q) const
  {
    return is_face ? grid_velocity_face_integrator.get_value(q) :
                     grid_velocity_cell_integrator.get_value(q);
  }

private:
  dealii::Tensor<2, dim, Number>
    to_dyadic(std::vector<dealii::Tensor<1, dim, Number>> const & vector_of_vectors)
  {
    auto tensor = dealii::Tensor<2, dim, Number>();
    for(unsigned int i = 0; i < dim; ++i)
      for(unsigned int j = 0; j < dim; ++j)
        tensor[i][j] = vector_of_vectors[i][j];

    return tensor;
  }

  dealii::Point<dim>
  batch_to_single_q_point(dealii::Point<dim, dealii::VectorizedArray<Number>> const & q_point_batch,
                          unsigned int const                                          v)
  {
    dealii::Point<dim> single_point;
    for(unsigned int d = 0; d < dim; ++d)
      single_point[d] = q_point_batch[d][v];

    return single_point;
  }

  dealii::Tensor<2, dim, Number> batch_to_single_dyadic(
    dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & dyadic_batch,
    unsigned int const                                              v)
  {
    dealii::Tensor<2, dim, Number> single_dyadic;
    for(unsigned int d1 = 0; d1 < dim; ++d1)
      for(unsigned int d2 = 0; d2 < dim; ++d2)
        single_dyadic[d1][d2] = dyadic_batch[d1][d2][v];

    return single_dyadic;
  }

  double
  compute_porosity(dealii::Point<dim> const &             q_point,
                   dealii::Tensor<2, dim, Number> const & def_gradient)
  {
    Number initial_porosity = initial_porosity_field->value(q_point);

    return 1.0 + (initial_porosity - 1.0) / determinant(def_gradient);
  }

private:
  lazy_ptr<VectorType> grid_coordinates_dof_vector;
  lazy_ptr<VectorType> grid_velocity_dof_vector;

  // IntegratorCell grid_coordinates_cell_integrator;

  dealii::FEValues<dim> grid_coordinates_cell_integrator;

  dealii::FEFaceValues<dim> grid_coordinates_face_integrator;

  std::vector<scalar> porosity_cell, porosity_face;

  IntegratorCell grid_velocity_cell_integrator;
  IntegratorFace grid_velocity_face_integrator;

  std::shared_ptr<dealii::Function<dim>> initial_porosity_field;
};

/*
template class StructureCouplingManager<2, float>;
template class StructureCouplingManager<2, double>;
template class StructureCouplingManager<3, float>;
template class StructureCouplingManager<3, double>;
 */
} // namespace Darcy
} // namespace ExaDG

#endif // EXADG_DARCY_GRID_VELOCITY_MANAGER_H
