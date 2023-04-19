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
#ifndef INCLUDE_EXADG_OPERATORS_VARIABLE_COEFFICIENTS_H_
#define INCLUDE_EXADG_OPERATORS_VARIABLE_COEFFICIENTS_H_

namespace ExaDG
{
template<typename coefficient_type>
class VariableCoefficientsCells
{
public:
  template<int dim, typename Number>
  void
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free,
             unsigned int const                      quad_index,
             coefficient_type const &                constant_coefficient)
  {
    reinit(matrix_free, quad_index);

    fill(constant_coefficient);
  }

  coefficient_type
  get_coefficient(unsigned int const cell, unsigned int const q) const
  {
    return coefficients_cell[cell][q];
  }

  void
  set_coefficient(unsigned int const cell, unsigned int const q, coefficient_type const & value)
  {
    coefficients_cell[cell][q] = value;
  }

private:
  template<int dim, typename Number>
  void
  reinit(dealii::MatrixFree<dim, Number> const & matrix_free, unsigned int const quad_index)
  {
    coefficients_cell.reinit(matrix_free.n_cell_batches(), matrix_free.get_n_q_points(quad_index));
  }

  void
  fill(coefficient_type const & constant_coefficient)
  {
    coefficients_cell.fill(constant_coefficient);
  }

  // variable coefficients
  dealii::Table<2, coefficient_type> coefficients_cell;
};

template<typename coefficient_type>
class VariableCoefficients
{
public:
  template<int dim, typename Number>
  void
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free,
             unsigned int const                      quad_index,
             bool const                              coefficients_differ_between_neighbors_in,
             bool const                              use_cell_based_face_loops_in)
  {
    coefficients_differ_between_neighbors = coefficients_differ_between_neighbors_in;
    use_cell_based_face_loops             = use_cell_based_face_loops_in;

    reinit(matrix_free, quad_index);
  }

  template<int dim, typename Number>
  void
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free,
             unsigned int const                      quad_index,
             coefficient_type const &                coefficient,
             bool const                              coefficients_differ_between_neighbors_in,
             bool const                              use_cell_based_face_loops_in)
  {
    initialize(matrix_free,
               quad_index,
               coefficients_differ_between_neighbors_in,
               use_cell_based_face_loops_in);

    fill(coefficient);
  }

  void
  set_coefficients(coefficient_type const & constant_coefficient)
  {
    fill(constant_coefficient);
  }

  coefficient_type
  get_coefficient_cell(unsigned int const cell, unsigned int const q) const
  {
    return coefficients_cell[cell][q];
  }

  void
  set_coefficient_cell(unsigned int const       cell,
                       unsigned int const       q,
                       coefficient_type const & value)
  {
    coefficients_cell[cell][q] = value;
  }

  coefficient_type
  get_coefficient_face(unsigned int const face, unsigned int const q) const
  {
    return coefficients_face[face][q];
  }

  void
  set_coefficient_face(unsigned int const       face,
                       unsigned int const       q,
                       coefficient_type const & value)
  {
    coefficients_face[face][q] = value;
  }

  coefficient_type
  get_coefficient_face_neighbor(unsigned int const face, unsigned int const q) const
  {
    return coefficients_face_neighbor[face][q];
  }

  void
  set_coefficient_face_neighbor(unsigned int const       face,
                                unsigned int const       q,
                                coefficient_type const & value)
  {
    coefficients_face_neighbor[face][q] = value;
  }

  coefficient_type
  get_coefficient_face_cell_based(unsigned int const face_index, unsigned int const q) const
  {
    return coefficients_face_cell_based[face_index][q];
  }

  void
  set_coefficient_face_cell_based(unsigned int const       cell_based_face_index,
                                  unsigned int const       q,
                                  coefficient_type const & value)
  {
    coefficients_face_cell_based[cell_based_face_index][q] = value;
  }

private:
  template<int dim, typename Number>
  void
  reinit(dealii::MatrixFree<dim, Number> const & matrix_free, unsigned int const quad_index)
  {
    coefficients_cell.reinit(matrix_free.n_cell_batches(), matrix_free.get_n_q_points(quad_index));

    coefficients_face.reinit(matrix_free.n_inner_face_batches() +
                               matrix_free.n_boundary_face_batches(),
                             matrix_free.get_n_q_points_face(quad_index));

    if(coefficients_differ_between_neighbors)
    {
      coefficients_face_neighbor.reinit(matrix_free.n_inner_face_batches(),
                                        matrix_free.get_n_q_points_face(quad_index));
    }

    if(use_cell_based_face_loops)
    {
      unsigned int const n_faces_per_cell =
        matrix_free.get_dof_handler().get_triangulation().get_reference_cells()[0].n_faces();

      coefficients_face_cell_based.reinit(matrix_free.n_cell_batches() * n_faces_per_cell,
                                          matrix_free.get_n_q_points(quad_index));
    }
  }

  void
  fill(coefficient_type const & constant_coefficient)
  {
    coefficients_cell.fill(constant_coefficient);
    coefficients_face.fill(constant_coefficient);

    if(coefficients_differ_between_neighbors)
      coefficients_face_neighbor.fill(constant_coefficient);

    if(use_cell_based_face_loops)
      coefficients_face_cell_based.fill(constant_coefficient);
  }

  // variable coefficients

  // cell
  dealii::Table<2, coefficient_type> coefficients_cell;

  // face-based loops
  dealii::Table<2, coefficient_type> coefficients_face;
  dealii::Table<2, coefficient_type> coefficients_face_neighbor;

  // cell-based face loops
  dealii::Table<2, coefficient_type> coefficients_face_cell_based;

  bool coefficients_differ_between_neighbors{false};
  bool use_cell_based_face_loops{false};
};

} // namespace ExaDG

#endif /* INCLUDE_EXADG_OPERATORS_VARIABLE_COEFFICIENTS_H_ */
