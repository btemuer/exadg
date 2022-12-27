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

#ifndef INCLUDE_EXADG_DARCY_SPATIAL_DISCRETIZATION_OPERATOR_COUPLED_H_
#define INCLUDE_EXADG_DARCY_SPATIAL_DISCRETIZATION_OPERATOR_COUPLED_H_

// deal.II
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/darcy/spatial_discretization/operators/permeability_operator.h>
#include <exadg/darcy/user_interface/field_functions.h>
#include <exadg/grid/grid.h>
#include <exadg/darcy/spatial_discretization/operators/divergence_operator.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/gradient_operator.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/momentum_operator.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/rhs_operator.h>
#include <exadg/incompressible_navier_stokes/user_interface/boundary_descriptor.h>
#include <exadg/incompressible_navier_stokes/user_interface/parameters.h>
#include <exadg/matrix_free/matrix_free_data.h>
#include <exadg/poisson/preconditioners/multigrid_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/preconditioner_base.h>
#include <exadg/solvers_and_preconditioners/solvers/iterative_solvers_dealii_wrapper.h>

namespace ExaDG
{
namespace Darcy
{
// forward declaration
template<int dim, typename Number>
class OperatorCoupled;

template<int dim, typename Number>
class LinearOperatorCoupled : public dealii::Subscriptor
{
private:
  using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<Number>;
  using PDEOperator     = OperatorCoupled<dim, Number>;

public:
  LinearOperatorCoupled() : dealii::Subscriptor(), pde_operator_(nullptr)
  {
  }

  void
  initialize(PDEOperator const & pde_operator)
  {
    this->pde_operator_ = &pde_operator;
  }

  /*
   * The implementation of linear solvers in deal.ii requires that a function called 'vmult' is
   * provided.
   */
  void
  vmult(BlockVectorType & dst, BlockVectorType const & src) const
  {
    pde_operator_->apply(dst, src);
  }

private:
  PDEOperator const * pde_operator_;
};

template<int dim, typename Number>
class BlockPreconditioner
{
private:
  using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<Number>;

  using PDEOperator = OperatorCoupled<dim, Number>;

public:
  BlockPreconditioner() : pde_operator(nullptr){};

  void
  initialize(PDEOperator * pde_operator_in)
  {
    pde_operator = pde_operator_in;
  }

  void
  update()
  {
    pde_operator->update_block_preconditioner();
  }

  void
  vmult(BlockVectorType & dst, BlockVectorType const & src) const
  {
    pde_operator->apply_block_preconditioner(dst, src);
  }

  std::shared_ptr<TimerTree>
  get_timings() const
  {
    AssertThrow(false,
                dealii::ExcMessage(
                  "Function get_timings() is not implemented for BlockPreconditioner."));

    return std::make_shared<TimerTree>();
  }

  PDEOperator * pde_operator;
};

template<int dim, typename Number = double>
class OperatorCoupled : public dealii::Subscriptor
{
private:
  using This = OperatorCoupled<dim, Number>;

  using VectorType      = dealii::LinearAlgebra::distributed::Vector<Number>;
  using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<Number>;

  using FaceIntegratorU = FaceIntegrator<dim, dim, Number>;
  using FaceIntegratorP = FaceIntegrator<dim, 1, Number>;

  using MultigridPoisson = Poisson::MultigridPreconditioner<dim, Number, 1>;

public:
  /*
   * Constructor
   */
  OperatorCoupled(std::shared_ptr<Grid<dim> const>                      grid,
                  std::shared_ptr<IncNS::BoundaryDescriptor<dim> const> boundary_descriptor,
                  std::shared_ptr<FieldFunctions<dim, Number> const>    field_functions_,
                  IncNS::Parameters const &                             parameters_,
                  std::string                                           field,
                  MPI_Comm const &                                      mpi_comm);

  void
  fill_matrix_free_data(MatrixFreeData<dim, Number> & matrix_free_data) const;

  void
  setup(std::shared_ptr<dealii::MatrixFree<dim, Number>> matrix_free,
        std::shared_ptr<MatrixFreeData<dim, Number>>     matrix_free_data);

  void
  setup_solvers();

  /*
   * Solve the problem
   */
  unsigned int
  solve(BlockVectorType & dst, BlockVectorType const & src, bool update_preconditioner);

  /*
   * Apply the matrix vector product
   */
  void
  apply(BlockVectorType & dst, BlockVectorType const & src) const;

  /*
   * Calculate the right-hand side
   */
  void
  rhs(BlockVectorType & dst) const;

  /*
   * Getters
   */

  dealii::MatrixFree<dim, Number> const &
  get_matrix_free() const;

  unsigned int
  get_dof_index_velocity() const;

  unsigned int
  get_dof_index_pressure() const;

  dealii::DoFHandler<dim> const &
  get_dof_handler_u() const;

  dealii::DoFHandler<dim> const &
  get_dof_handler_p() const;

  unsigned int
  get_quad_index_velocity() const;

  unsigned int
  get_quad_index_pressure() const;

  std::shared_ptr<dealii::Mapping<dim> const>
  get_mapping() const;

  dealii::types::global_dof_index
  get_number_of_dofs() const;

  /*
   * Initialization of vectors
   */
  void
  initialize_vector_pressure(VectorType & src) const;

  void
  initialize_vector_velocity(VectorType & src) const;

  void
  initialize_block_vector_velocity_pressure(BlockVectorType & src) const;

  /*
   * Prescribe initial conditions using a specified analytical/initial solution function.
   */
  void
  prescribe_initial_conditions(VectorType & velocity, VectorType & pressure) const;

  /*
   * Preconditioner interface
   */
  void
  update_block_preconditioner();

  void
  apply_block_preconditioner(BlockVectorType & dst, BlockVectorType const & src) const;

private:
  void
  initialize_solver();

  void
  initialize_block_preconditioner();

  void
  initialize_temporary_vectors();

  /*
   * Velocity / Momentum block
   */
  void
  initialize_preconditioner_velocity_block();

  void
  apply_preconditioner_velocity_block(VectorType & dst, VectorType const & src) const;

  /*
   * Pressure / Schur-complement block
   */
  void
  initialize_preconditioner_pressure_block();

  void
  apply_preconditioner_pressure_block(VectorType & dst, VectorType const & src) const;

  void
  setup_multigrid_preconditioner_pressure_block();

  void
  initialize_boundary_descriptor_laplace();

  void
  distribute_dofs();

  void
  initialize_operators();


  /*
   * Grid
   */
  std::shared_ptr<Grid<dim> const> grid_;

  /*
   * User interface: Boundary conditions and field functions
   */
  std::shared_ptr<IncNS::BoundaryDescriptor<dim> const> boundary_descriptor_;
  std::shared_ptr<FieldFunctions<dim, Number> const>    field_functions_;

  /*
   * List of parameters
   */
  IncNS::Parameters const & param_;

  /*
   * A name describing the field being solved
   */
  std::string const field_;

  /*
   * Basic finite element ingredients
   */
  std::shared_ptr<dealii::FiniteElement<dim>> fe_u_;
  dealii::FE_DGQ<dim>                         fe_p_;

  dealii::DoFHandler<dim> dof_handler_u_;
  dealii::DoFHandler<dim> dof_handler_p_;

  dealii::AffineConstraints<Number> constraint_u_;
  dealii::AffineConstraints<Number> constraint_p_;

  std::string const dof_index_u_ = "velocity";
  std::string const dof_index_p_ = "pressure";

  std::string const quad_index_u_               = "velocity";
  std::string const quad_index_p_               = "pressure";
  std::string const quad_index_u_gauss_lobatto_ = "velocity_gauss_lobatto";
  std::string const quad_index_p_gauss_lobatto_ = "pressure_gauss_lobatto";

  std::shared_ptr<MatrixFreeData<dim, Number>>     matrix_free_data_;
  std::shared_ptr<dealii::MatrixFree<dim, Number>> matrix_free_;

  /*
   * Basic operators
   */
  PermeabilityOperator<dim, Number>    permeability_operator_;
  DivergenceOperator<dim, Number>      divergence_operator_;
  IncNS::RHSOperator<dim, Number>      rhs_operator_;
  IncNS::GradientOperator<dim, Number> gradient_operator_;

  MPI_Comm const mpi_comm_;

  dealii::ConditionalOStream pcout_;

  /*
   * Linear operator
   */
  LinearOperatorCoupled<dim, Number> linear_operator_;

  /*
   * Linear solver
   */
  std::shared_ptr<Krylov::SolverBase<BlockVectorType>> linear_solver_;

  /*
   * Block preconditioner instance (aggregation of the velocity
   * and the pressure block preconditioners
   */
  using Preconditioner = BlockPreconditioner<dim, Number>;
  Preconditioner block_preconditioner_;

  /*
   * Preconditioner and solver for the velocity / momentum block
   */
  std::shared_ptr<PreconditionerBase<Number>> preconditioner_velocity_block_;

  /*
   * Preconditioner and solver for the pressure / Schur-complement block
   */
  std::shared_ptr<PreconditionerBase<Number>> preconditioner_pressure_block_;

  /*
   * Used for the block preconditioner (or more precisely for the Schur-complement preconditioner
   * and the preconditioner used to approximately invert the Laplace operator)
   *
   * The functions specified in BoundaryDescriptorLaplace are irrelevant for a coupled solution
   * approach (since the pressure Poisson operator is only needed for preconditioning, and hence,
   * only the homogeneous part of the operator has to be evaluated so that the boundary conditions
   * are never applied).
   */
  std::shared_ptr<Poisson::BoundaryDescriptor<0, dim>> boundary_descriptor_laplace_;

  // temporary vectors that are necessary when using preconditioners of block-triangular type
  VectorType mutable vec_tmp_pressure_;
  VectorType mutable vec_tmp_velocity_, vec_tmp_velocity_2_;
};

} // namespace Darcy
} // namespace ExaDG
#endif /* INCLUDE_EXADG_DARCY_SPATIAL_DISCRETIZATION_OPERATOR_COUPLED_H_ \
        */
