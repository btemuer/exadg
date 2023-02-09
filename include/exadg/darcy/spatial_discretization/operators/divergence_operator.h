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

#include <exadg/darcy/spatial_discretization/structure_coupling_manager.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/weak_boundary_conditions.h>
#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/mapping_flags.h>


#ifndef EXADG_DARCY_DIVERGENCE_OPERATOR_H
#  define EXADG_DARCY_DIVERGENCE_OPERATOR_H

namespace ExaDG
{
namespace Darcy
{
namespace Operators
{
template<int dim>
struct DivergenceKernelData
{
  std::shared_ptr<dealii::Function<dim>> initial_porosity_field;
};

template<int dim, typename Number>
class DivergenceKernel
{
private:
  using scalar = dealii::VectorizedArray<Number>;
  using vector = dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>;

public:
  void
  reinit(DivergenceKernelData<dim> const & data_in) const
  {
    data = data_in;
  }

  static MappingFlags
  get_mapping_flags()
  {
    MappingFlags flags;

    flags.cells =
      dealii::update_JxW_values | dealii::update_gradients | dealii::update_quadrature_points;
    flags.inner_faces =
      dealii::update_JxW_values | dealii::update_normal_vectors | dealii::update_quadrature_points;
    flags.boundary_faces =
      dealii::update_JxW_values | dealii::update_normal_vectors | dealii::update_quadrature_points;

    return flags;
  }

  /*
   *  This function implements the central flux as the numerical flux function.
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_flux(vector const & velocity_value_m,
                   vector const & velocity_value_p,
                   scalar const & fac) const
  {
    // fac is either (porosity) or (1 - porosity)
    return 0.5 * fac * (velocity_value_m + velocity_value_p);
  }

  /*
   * Volume flux, i.e., the term occurring in the volume integral for
   * the weak formulation (performing integration-by-parts)
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    get_volume_flux(vector const & velocity_value, scalar const fac) const
  {
    // minus sign due to integration by parts
    return -fac * velocity_value;
  }



private:
  mutable DivergenceKernelData<dim> data;
};
} // namespace Operators

template<int dim>
struct DivergenceOperatorData
{
  unsigned int dof_index_velocity{0};
  unsigned int dof_index_pressure{1};

  unsigned int quad_index{0};

  bool use_boundary_data{true};

  bool ale{false};

  std::shared_ptr<IncNS::BoundaryDescriptorU<dim> const> bc{};

  Operators::DivergenceKernelData<dim> kernel_data{};
};

template<int dim, typename Number>
class DivergenceOperator
{
private:
  using This = DivergenceOperator<dim, Number>;

  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;

  using scalar = dealii::VectorizedArray<Number>;
  using vector = dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>;

  using point = dealii::Point<dim, dealii::VectorizedArray<Number>>;

  using Range = std::pair<unsigned int, unsigned int>;

  using CellIntegratorU = CellIntegrator<dim, dim, Number>;
  using CellIntegratorP = CellIntegrator<dim, 1, Number>;
  using FaceIntegratorU = FaceIntegrator<dim, dim, Number>;
  using FaceIntegratorP = FaceIntegrator<dim, 1, Number>;

public:
  DivergenceOperator();

  void
  initialize(dealii::MatrixFree<dim, Number> const &                matrix_free,
             DivergenceOperatorData<dim> const &                    data,
             std::shared_ptr<StructureCouplingManager<dim, Number>> grid_velocity_manager_in);

  // homogeneous operator
  void
  apply(VectorType & dst, VectorType const & src) const;

  void
  apply_add(VectorType & dst, VectorType const & src) const;

  // inhomogeneous operator
  void
  rhs(VectorType & dst, Number const evaluation_time) const;

  void
  rhs_add(VectorType & dst, Number const evaluation_time) const;

private:
  template<typename Integrator>
  scalar
  calculate_porosity(bool const is_face, Integrator const & integrator, unsigned int const q) const;

  void
  cell_loop_hom_operator(dealii::MatrixFree<dim, Number> const & matrix_free,
                         VectorType &                            dst,
                         VectorType const &                      src,
                         Range const &                           cell_range) const;

  void
  face_loop_hom_operator(dealii::MatrixFree<dim, Number> const & matrix_free,
                         VectorType &                            dst,
                         VectorType const &                      src,
                         Range const &                           face_range) const;

  void
  boundary_face_loop_hom_operator(dealii::MatrixFree<dim, Number> const & matrix_free,
                                  VectorType &                            dst,
                                  VectorType const &                      src,
                                  Range const &                           face_range) const;

  void
  cell_loop_inhom_operator(dealii::MatrixFree<dim, Number> const & matrix_free,
                           VectorType &                            dst,
                           VectorType const &                      src,
                           Range const &                           cell_range) const;

  void
  face_loop_inhom_operator(dealii::MatrixFree<dim, Number> const & matrix_free,
                           VectorType &                            dst,
                           VectorType const &                      src,
                           Range const &                           face_range) const;

  void
  boundary_face_loop_inhom_operator(dealii::MatrixFree<dim, Number> const &       matrix_free,
                                    VectorType &                                  dst,
                                    VectorType const &                            src,
                                    std::pair<unsigned int, unsigned int> const & face_range) const;

  dealii::MatrixFree<dim, Number> const * matrix_free;

  DivergenceOperatorData<dim> data;

  Operators::DivergenceKernel<dim, Number> kernel;

  mutable double time{};

  // needed if Dirichlet boundary condition is evaluated from dof vector
  mutable VectorType const * velocity_bc;

  std::shared_ptr<StructureCouplingManager<dim, Number>> structure_coupling_manager;
};



} // namespace Darcy
} // namespace ExaDG

#endif // EXADG_DARCY_DIVERGENCE_OPERATOR_H
