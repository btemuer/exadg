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
private:
  using coefficient_function_type =
    std::function<coefficient_type(unsigned int const, unsigned int const)>;

  using cell_based_coefficient_function_type =
    std::function<coefficient_type(unsigned int const, unsigned int const, unsigned int const)>;

public:
  template<int dim, typename Number>
  void
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free,
             unsigned int const                      quad_index,
             coefficient_type const &                coefficient,
             bool const                              coefficients_differ_between_neighbors_in,
             bool const                              cell_based_face_coefficient_function_in)
  {
    coefficients_differ_between_neighbors = coefficients_differ_between_neighbors_in;
    cell_based_face_loop                  = cell_based_face_coefficient_function_in;

    reinit(matrix_free, quad_index);

    fill(coefficient);
  }

  template<int dim, typename Number>
  void
  initialize(dealii::MatrixFree<dim, Number> const &      matrix_free,
             unsigned int const                           quad_index,
             coefficient_function_type const &            cell_coefficient_function,
             coefficient_function_type const &            face_coefficient_function,
             coefficient_function_type const &            neighbor_face_coefficient_function   = {},
             cell_based_coefficient_function_type const & cell_based_face_coefficient_function = {})
  {
    if(neighbor_face_coefficient_function)
      coefficients_differ_between_neighbors = true;

    if(cell_based_face_coefficient_function)
      cell_based_face_loop = true;

    reinit(matrix_free, quad_index);

    fill(cell_coefficient_function,
         face_coefficient_function,
         neighbor_face_coefficient_function,
         cell_based_face_coefficient_function);
  }

  void
  set_coefficients(coefficient_type const & constant_coefficient)
  {
    fill(constant_coefficient);
  }

  void
  set_coefficients(
    coefficient_function_type const &            cell_coefficient_function,
    coefficient_function_type const &            face_coefficient_function,
    coefficient_function_type const &            neighbor_face_coefficient_function   = {},
    cell_based_coefficient_function_type const & cell_based_face_coefficient_function = {})
  {
    fill(cell_coefficient_function,
         face_coefficient_function,
         neighbor_face_coefficient_function,
         cell_based_face_coefficient_function);
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
  get_coefficient_cell_based_face(unsigned int const cell,
                                  unsigned int const face,
                                  unsigned int const q) const
  {
    return coefficients_face_cell_based[cell][face][q];
  }

  void
  set_coefficient_cell_based_face(unsigned int const       cell,
                                  unsigned int const       face,
                                  unsigned int const       q,
                                  coefficient_type const & value)
  {
    coefficients_face_cell_based[cell][face][q] = value;
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

    if(cell_based_face_loop)
    {
      coefficients_face_cell_based.reinit(
        matrix_free.n_cell_batches(),
        matrix_free.get_mapping_info().reference_cell_types[0][0].n_faces(),
        matrix_free.get_n_q_points(quad_index));
    }
  }

  void
  fill(coefficient_function_type const &            cell_coefficient_function,
       coefficient_function_type const &            face_coefficient_function,
       coefficient_function_type const &            neighbor_face_coefficient_function,
       cell_based_coefficient_function_type const & cell_based_face_coefficient_function)
  {
    if(cell_based_face_loop)
      Assert(cell_based_face_coefficient_function,
             dealii::ExcMessage("Cell based face coefficient function must be specified."));

    if(coefficients_differ_between_neighbors)
      Assert(neighbor_face_coefficient_function,
             dealii::ExcMessage("Neighbor face coefficient function must be specified."));

    unsigned int const n_cells = coefficients_cell.size(0);
    unsigned int const n_faces = coefficients_face.size(0);

    unsigned int const n_faces_per_cell =
      cell_based_face_loop ? coefficients_face_cell_based.size(1) : 0;

    unsigned int const n_cell_q_points = coefficients_cell.size(1);
    unsigned int const n_face_q_points = coefficients_face.size(1);

    for(unsigned int cell = 0; cell < n_cells; ++cell)
    {
      for(unsigned int q = 0; q < n_cell_q_points; ++q)
        set_coefficient_cell(cell, q, cell_coefficient_function(cell, q));

      for(unsigned int face = 0; face < n_faces_per_cell; ++face)
        for(unsigned int q = 0; q < n_cell_q_points; ++q)
          set_coefficient_cell_based_face(cell,
                                          face,
                                          q,
                                          cell_based_face_coefficient_function(cell, face, q));
    }

    for(unsigned int face = 0; face < n_faces; ++face)
    {
      for(unsigned int q = 0; q < n_face_q_points; ++q)
      {
        set_coefficient_face(face, q, face_coefficient_function(face, q));
        if(coefficients_differ_between_neighbors)
          set_coefficient_face_neighbor(face, q, neighbor_face_coefficient_function(face, q));
      }
    }
  }

  void
  fill(coefficient_type const & constant_coefficient)
  {
    coefficients_cell.fill(constant_coefficient);
    coefficients_face.fill(constant_coefficient);

    if(coefficients_differ_between_neighbors)
      coefficients_face_neighbor.fill(constant_coefficient);

    if(cell_based_face_loop)
      coefficients_face_cell_based.fill(constant_coefficient);
  }

  // variable coefficients

  // cell
  dealii::Table<2, coefficient_type> coefficients_cell;

  // face-based loops
  dealii::Table<2, coefficient_type> coefficients_face;
  dealii::Table<2, coefficient_type> coefficients_face_neighbor;

  // cell-based face loops
  dealii::Table<3, coefficient_type> coefficients_face_cell_based;

  bool coefficients_differ_between_neighbors{false};
  bool cell_based_face_loop{false};
};

} // namespace ExaDG

#endif /* INCLUDE_EXADG_OPERATORS_VARIABLE_COEFFICIENTS_H_ */
