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
  template<int dim, typename Number, typename T>
  void
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free,
             unsigned int const                      quad_index,
             T const &                               coefficient)
  {
    reinit(matrix_free, quad_index);

    fill(coefficient);
  }

  template<typename F>
  void
  set_cofficients(F const & coefficient_function)
  {
    fill(coefficient_function);
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

  // TODO
  //
  //  coefficient_type
  //  get_coefficient_cell_based(unsigned int const face,
  //                             unsigned int const q) const
  //  {
  //    return coefficients_face_cell_based[face][q];
  //  }
  //
  //  void
  //  set_coefficient_cell_based(unsigned int const       face,
  //                             unsigned int const       q,
  //                             coefficient_type const & value)
  //  {
  //    coefficients_face_cell_based[face][q] = value;
  //  }

private:
  template<int dim, typename Number>
  void
  reinit(dealii::MatrixFree<dim, Number> const & matrix_free, unsigned int const quad_index)
  {
    coefficients_cell.reinit(matrix_free.n_cell_batches(), matrix_free.get_n_q_points(quad_index));

    coefficients_face.reinit(matrix_free.n_inner_face_batches() +
                               matrix_free.n_boundary_face_batches(),
                             matrix_free.get_n_q_points_face(quad_index));

    coefficients_face_neighbor.reinit(matrix_free.n_inner_face_batches(),
                                      matrix_free.get_n_q_points_face(quad_index));

    // TODO
    // // cell-based face loops
    // coefficients_face_cell_based.reinit(matrix_free.n_cell_batches()*2*dim,
    // matrix_free.get_n_q_points_face(quad_index));
  }

  void
  fill(std::function<coefficient_type(unsigned int const, unsigned int const)> const &
         coefficient_function)
  {
    unsigned int const n_cells = coefficients_cell.size(0);
    unsigned int const n_faces = coefficients_face.size(0);

    unsigned int const n_cell_q_points = coefficients_cell.size(1);
    unsigned int const n_face_q_points = coefficients_face.size(1);

    for(unsigned int cell = 0; cell < n_cells; ++cell)
      for(unsigned int q = 0; q < n_cell_q_points; ++q)
        set_coefficient_cell(cell, q, coefficient_function(cell, q));

    for(unsigned int face = 0; face < n_faces; ++face)
      for(unsigned int q = 0; q < n_face_q_points; ++q)
        set_coefficient_face(face, q, coefficient_function(face, q));
  }

  void
  fill(coefficient_type const & constant_coefficient)
  {
    coefficients_cell.fill(constant_coefficient);
    coefficients_face.fill(constant_coefficient);
    coefficients_face_neighbor.fill(constant_coefficient);

    // TODO
    // coefficients_face_cell_based.fill(constant_coefficient);
  }

  // variable coefficients

  // cell
  dealii::Table<2, coefficient_type> coefficients_cell;

  // face-based loops
  dealii::Table<2, coefficient_type> coefficients_face;
  dealii::Table<2, coefficient_type> coefficients_face_neighbor;

  // TODO
  //  // cell-based face loops
  //  dealii::Table<2, coefficient_type> coefficients_face_cell_based;
};

} // namespace ExaDG

#endif /* INCLUDE_EXADG_OPERATORS_VARIABLE_COEFFICIENTS_H_ */
