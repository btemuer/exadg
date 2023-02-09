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

#ifndef EXADG_DARCY_COUPLING_MOMENTUM_OPERATOR_H
#define EXADG_DARCY_COUPLING_MOMENTUM_OPERATOR_H

#include "exadg/darcy/user_interface/parameters.h"
#include "exadg/matrix_free/integrators.h"
#include "exadg/operators/operator_base.h"

namespace ExaDG
{
namespace Darcy
{
namespace Operators
{
struct GridConvectionKernelData
{
  double upwind_factor{1.0};
};

template<int dim, typename Number>
class GridConvectionKernel
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
  reinit(GridConvectionKernelData const & data_in)
  {
    data = data_in;
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

  inline DEAL_II_ALWAYS_INLINE //
    vector
    get_volume_flux(dyadic const & velocity_gradient, vector const & grid_velocity_value)
  {
    return velocity_gradient * grid_velocity_value;
  }

  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_flux(vector const & velocity_value_m,
                   vector const & velocity_value_p,
                   vector const & grid_velocity_value_m,
                   vector const & normal_m)
  {
    vector const grid_velocity_value  = grid_velocity_value_m;
    scalar const normal_grid_velocity = grid_velocity_value * normal_m;

    vector const average_velocity = 0.5 * (velocity_value_m + velocity_value_p);
    vector const velocity_jump    = velocity_value_m - velocity_value_p;

    return normal_grid_velocity * (average_velocity - velocity_value_m) +
           0.5 * data.upwind_factor * std::abs(normal_grid_velocity) * velocity_jump;
  }

private:
  GridConvectionKernelData data;
};

} // namespace Operators
} // namespace Darcy
} // namespace ExaDG

#endif // EXADG_DARCY_COUPLING_MOMENTUM_OPERATOR_H
