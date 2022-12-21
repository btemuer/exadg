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

// deal.II
#include <deal.II/numerics/vector_tools.h>

// ExaDG
#include <exadg/darcy/spatial_discretization/operator_coupled.h>
#include <exadg/poisson/spatial_discretization/laplace_operator.h>
#include <exadg/solvers_and_preconditioners/preconditioners/inverse_mass_preconditioner.h>

namespace ExaDG
{
namespace Darcy
{
template<int dim, typename Number>
OperatorCoupled<dim, Number>::OperatorCoupled(
  std::shared_ptr<Grid<dim> const>                      grid,
  std::shared_ptr<IncNS::BoundaryDescriptor<dim> const> boundary_descriptor,
  std::shared_ptr<IncNS::FieldFunctions<dim> const>     field_functions,
  IncNS::Parameters const &                             parameters,
  std::string                                           field,
  MPI_Comm const &                                      mpi_comm)
  : dealii::Subscriptor(),
    grid_(grid),
    boundary_descriptor_(boundary_descriptor),
    field_functions_(field_functions),
    param_(parameters),
    field_(field),
    fe_p_(parameters.get_degree_p(parameters.degree_u)),
    dof_handler_u_(*(grid->triangulation)),
    dof_handler_p_(*(grid->triangulation)),
    mpi_comm_(mpi_comm),
    pcout_(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
{
  pcout_ << std::endl << "Construct Darcy operator ..." << std::endl << std::flush;

  initialize_boundary_descriptor_laplace();

  distribute_dofs();

  constraint_u_.close();
  constraint_p_.close();

  // Erroneously, the boundary descriptor might contain too many boundary IDs which
  // do not even exist in the triangulation. Here, we make sure that each entry of
  // the boundary descriptor has indeed a counterpart in the triangulation.
  std::vector<dealii::types::boundary_id> boundary_ids = grid->triangulation->get_boundary_ids();
  for(auto it = boundary_descriptor->pressure->dirichlet_bc.begin();
      it != boundary_descriptor->pressure->dirichlet_bc.end();
      ++it)
  {
    bool const triangulation_has_boundary_id =
      std::find(boundary_ids.begin(), boundary_ids.end(), it->first) != boundary_ids.end();

    AssertThrow(triangulation_has_boundary_id,
                dealii::ExcMessage("The boundary descriptor for the pressure contains boundary IDs "
                                   "that are not part of the triangulation."));
  }

  pcout_ << std::endl << "... done!" << std::endl << std::flush;
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::fill_matrix_free_data(
  MatrixFreeData<dim, Number> & matrix_free_data) const
{
  matrix_free_data.append_mapping_flags(MassKernel<dim, Number>::get_mapping_flags());
  matrix_free_data.append_mapping_flags(
    IncNS::Operators::DivergenceKernel<dim, Number>::get_mapping_flags());
  matrix_free_data.append_mapping_flags(
    IncNS::Operators::GradientKernel<dim, Number>::get_mapping_flags());

  if(param_.right_hand_side)
    matrix_free_data.append_mapping_flags(
      IncNS::Operators::RHSKernel<dim, Number>::get_mapping_flags());

  // DoF handler
  {
    matrix_free_data.insert_dof_handler(&dof_handler_u_, field_ + dof_index_u_);
    matrix_free_data.insert_dof_handler(&dof_handler_p_, field_ + dof_index_p_);
  }

  // Constraints
  {
    matrix_free_data.insert_constraint(&constraint_u_, field_ + dof_index_u_);
    matrix_free_data.insert_constraint(&constraint_p_, field_ + dof_index_p_);
  }

  // Quadrature
  {
    matrix_free_data.insert_quadrature(dealii::QGauss<1>(param_.degree_u + 1),
                                       field_ + quad_index_u_);
    matrix_free_data.insert_quadrature(dealii::QGauss<1>(param_.get_degree_p(param_.degree_u) + 1),
                                       field_ + quad_index_p_);
    matrix_free_data.insert_quadrature(dealii::QGaussLobatto<1>(param_.degree_u + 1),
                                       field_ + quad_index_u_gauss_lobatto_);
    matrix_free_data.insert_quadrature(dealii::QGaussLobatto<1>(
                                         param_.get_degree_p(param_.degree_u) + 1),
                                       field_ + quad_index_p_gauss_lobatto_);
  }
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::setup(
  std::shared_ptr<dealii::MatrixFree<dim, Number>> matrix_free_in,
  std::shared_ptr<MatrixFreeData<dim, Number>>     matrix_free_data_in)
{
  pcout_ << std::endl << "Setup Darcy operator ..." << std::endl << std::flush;

  matrix_free_      = matrix_free_in;
  matrix_free_data_ = matrix_free_data_in;

  initialize_operators();

  pcout_ << std::endl << "... done!" << std::endl << std::flush;
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::setup_solvers(VectorType const & velocity)
{
  pcout_ << std::endl << "Setup Darcy solver ..." << std::endl;

  momentum_operator_.set_scaling_factor_mass_operator(1.0);
  momentum_operator_.set_velocity_ptr(velocity);

  initialize_block_preconditioner();

  initialize_solver();

  pcout_ << std::endl << "... done!" << std::endl;
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::initialize_solver()
{
  linear_operator_.initialize(*this);

  // setup linear solver
  if(param_.solver_coupled == IncNS::SolverCoupled::GMRES)
  {
    Krylov::SolverDataGMRES solver_data;
    solver_data.max_iter             = this->param_.solver_data_coupled.max_iter;
    solver_data.solver_tolerance_abs = this->param_.solver_data_coupled.abs_tol;
    solver_data.solver_tolerance_rel = this->param_.solver_data_coupled.rel_tol;
    solver_data.max_n_tmp_vectors    = this->param_.solver_data_coupled.max_krylov_size;
    solver_data.compute_eigenvalues  = false;

    if(this->param_.preconditioner_coupled != IncNS::PreconditionerCoupled::None)
      solver_data.use_preconditioner = true;

    linear_solver_ = std::make_shared<
      Krylov::SolverGMRES<LinearOperatorCoupled<dim, Number>, Preconditioner, BlockVectorType>>(
      linear_operator_, block_preconditioner_, solver_data, mpi_comm_);
  }
  else if(param_.solver_coupled == IncNS::SolverCoupled::FGMRES)
  {
    Krylov::SolverDataFGMRES solver_data;
    solver_data.max_iter             = param_.solver_data_coupled.max_iter;
    solver_data.solver_tolerance_abs = param_.solver_data_coupled.abs_tol;
    solver_data.solver_tolerance_rel = param_.solver_data_coupled.rel_tol;
    solver_data.max_n_tmp_vectors    = param_.solver_data_coupled.max_krylov_size;

    if(param_.preconditioner_coupled != IncNS::PreconditionerCoupled::None)
      solver_data.use_preconditioner = true;

    linear_solver_ = std::make_shared<
      Krylov::SolverFGMRES<LinearOperatorCoupled<dim, Number>, Preconditioner, BlockVectorType>>(
      linear_operator_, block_preconditioner_, solver_data);
  }
  else
    AssertThrow(false, dealii::ExcMessage("Specified solver is not implemented."));
}

template<int dim, typename Number>
unsigned int
OperatorCoupled<dim, Number>::solve(BlockVectorType &       dst,
                                    BlockVectorType const & src,
                                    bool const              update_preconditioner)
{
  return linear_solver_->solve(dst, src, update_preconditioner);
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::apply(OperatorCoupled::BlockVectorType &       dst,
                                    OperatorCoupled::BlockVectorType const & src) const
{
  // (0,0) block of the matrix
  {
    mass_operator_.apply_scale(dst.block(0), 1.0, src.block(0));
  }

  // (0,1) block of the matrix
  // gradient operator: dst = velocity, src = pressure
  {
    gradient_operator_.apply_add(dst.block(0), src.block(1));
  }

  // (1,0) block of the matrix
  // divergence operator: dst = pressure, src = velocity
  {
    divergence_operator_.apply(dst.block(1), src.block(0));
    dst.block(1) *= -1.0;
  }
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::rhs(OperatorCoupled::BlockVectorType & dst) const
{
  // Velocity block
  {
    gradient_operator_.rhs(dst.block(0), 0.0);
    // gradient_operator_.rhs_add(dst.block(0), 0.0);

    if(param_.right_hand_side)
      rhs_operator_.evaluate_add(dst.block(0), 0.0);
  }

  // Pressure block
  {
    divergence_operator_.rhs(dst.block(1), 0.0);
    dst.block(1) *= -1.0;
  }
}

template<int dim, typename Number>
dealii::MatrixFree<dim, Number> const &
OperatorCoupled<dim, Number>::get_matrix_free() const
{
  return *matrix_free_;
}

template<int dim, typename Number>
unsigned int
OperatorCoupled<dim, Number>::get_dof_index_velocity() const
{
  return matrix_free_data_->get_dof_index(field_ + dof_index_u_);
}

template<int dim, typename Number>
unsigned int
OperatorCoupled<dim, Number>::get_dof_index_pressure() const
{
  return matrix_free_data_->get_dof_index(field_ + dof_index_p_);
}

template<int dim, typename Number>
dealii::DoFHandler<dim> const &
OperatorCoupled<dim, Number>::get_dof_handler_u() const
{
  return dof_handler_u_;
}

template<int dim, typename Number>
dealii::DoFHandler<dim> const &
OperatorCoupled<dim, Number>::get_dof_handler_p() const
{
  return dof_handler_p_;
}


template<int dim, typename Number>
unsigned int
OperatorCoupled<dim, Number>::get_quad_index_velocity() const
{
  return matrix_free_data_->get_quad_index(field_ + quad_index_u_);
}

template<int dim, typename Number>
unsigned int
OperatorCoupled<dim, Number>::get_quad_index_pressure() const
{
  return matrix_free_data_->get_quad_index(field_ + quad_index_p_);
}

template<int dim, typename Number>
std::shared_ptr<dealii::Mapping<dim> const>
OperatorCoupled<dim, Number>::get_mapping() const
{
  return grid_->get_mapping();
}

template<int dim, typename Number>
dealii::types::global_dof_index
OperatorCoupled<dim, Number>::get_number_of_dofs() const
{
  return dof_handler_u_.n_dofs() + dof_handler_p_.n_dofs();
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::initialize_vector_pressure(VectorType & src) const
{
  matrix_free_->initialize_dof_vector(src, get_dof_index_pressure());
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::initialize_vector_velocity(VectorType & src) const
{
  matrix_free_->initialize_dof_vector(src, get_dof_index_velocity());
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::initialize_block_vector_velocity_pressure(BlockVectorType & src) const
{
  // Velocity (0th block) + Pressure (1th block)
  src.reinit(2);

  matrix_free_->initialize_dof_vector(src.block(0), get_dof_index_velocity());
  matrix_free_->initialize_dof_vector(src.block(1), get_dof_index_pressure());

  src.collect_sizes();
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::prescribe_initial_conditions(
  OperatorCoupled::VectorType & velocity,
  OperatorCoupled::VectorType & pressure) const
{
  field_functions_->initial_solution_velocity->set_time(0.0);
  field_functions_->initial_solution_pressure->set_time(0.0);

  typedef dealii::LinearAlgebra::distributed::Vector<double> VectorTypeDouble;

  VectorTypeDouble velocity_double;
  VectorTypeDouble pressure_double;
  velocity_double = velocity;
  pressure_double = pressure;

  // Initial guess is a zero field for the velocity and the pressure
  dealii::VectorTools::interpolate(*get_mapping(),
                                   dof_handler_u_,
                                   *(field_functions_->initial_solution_velocity),
                                   velocity_double);

  dealii::VectorTools::interpolate(*get_mapping(),
                                   dof_handler_p_,
                                   *(field_functions_->initial_solution_pressure),
                                   pressure_double);

  velocity = velocity_double;
  pressure = pressure_double;
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::update_block_preconditioner()
{
  // Velocity / momentum block
  preconditioner_velocity_block_->update();

  // Pressure / Schur-complement block
  // Do nothing (no ALE)...
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::apply_block_preconditioner(
  OperatorCoupled::BlockVectorType &       dst,
  OperatorCoupled::BlockVectorType const & src) const
{
  auto const type = param_.preconditioner_coupled;

  if(type == IncNS::PreconditionerCoupled::BlockDiagonal)
  {
    /*                        / A^{-1}   0    \   / A^{-1}  0 \   / I      0    \
     *   -> P_diagonal^{-1} = |               | = |           | * |             |
     *                        \   0   -S^{-1} /   \   0     I /   \ 0   -S^{-1} /
     */

    // Preconditioner for the pressure / Schur-complement block

    /*
     *         / I      0    \
     *  temp = |             | * src
     *         \ 0   -S^{-1} /
     */
    {
      apply_preconditioner_pressure_block(dst.block(1), src.block(1));
    }

    // Preconditioner for the velocity / momentum block
    /*
     *        / A^{-1}  0 \
     *  dst = |           | * temp
     *        \   0     I /
     */
    {
      apply_preconditioner_velocity_block(dst.block(0), src.block(0));
    }
  }
  else if(type == IncNS::PreconditionerCoupled::BlockTriangular)
  {
    /*
     *                         / A^{-1}  0 \   / I  B^{T} \   / I      0    \
     *  -> P_triangular^{-1} = |           | * |          | * |             |
     *                         \   0     I /   \ 0   -I   /   \ 0   -S^{-1} /
     */

    /*
     *        / I      0    \
     *  dst = |             | * src
     *        \ 0   -S^{-1} /
     */
    {
      // For the velocity block simply copy data from src to dst
      dst.block(0) = src.block(0);
      // Apply preconditioner for the pressure/Schur-complement block
      apply_preconditioner_pressure_block(dst.block(1), src.block(1));
    }

    /*
     *        / I  B^{T} \
     *  dst = |          | * dst
     *        \ 0   -I   /
     */
    {
      // Apply gradient operator and add to dst vector.
      gradient_operator_.apply_add(dst.block(0), dst.block(1));
      dst.block(1) *= -1.0;
    }

    /*
     *        / A^{-1}  0 \
     *  dst = |           | * dst
     *        \   0     I /
     */
    {
      // Copy data from dst.block(0) to vec_tmp_velocity before
      // applying the preconditioner for the velocity block.
      vec_tmp_velocity_ = dst.block(0);
      // Apply preconditioner for velocity/momentum block.
      apply_preconditioner_velocity_block(dst.block(0), vec_tmp_velocity_);
    }
  }
  else if(type == IncNS::PreconditionerCoupled::BlockTriangularFactorization)
  {
    /*
     *                          / I  - A^{-1} B^{T} \   / I      0    \   / I   0 \   / A^{-1} 0 \
     *  -> P_tria-factor^{-1} = |                   | * |             | * |       | * |          |
     *                          \ 0          I      /   \ 0   -S^{-1} /   \ B  -I /   \   0    I /
     */

    /*
     *        / A^{-1}  0 \
     *  dst = |           | * src
     *        \   0     I /
     */
    {
      // For the pressure block simply copy data from src to dst
      dst.block(1) = src.block(1);
      // Apply preconditioner for velocity/momentum block
      apply_preconditioner_velocity_block(dst.block(0), src.block(0));
    }

    /*
     *        / I   0 \
     *  dst = |       | * dst
     *        \ B  -I /
     */
    {
      // Note that B represents NEGATIVE divergence operator, i.e.,
      // applying -B is equal to applying the divergence operator
      divergence_operator_.apply_add(dst.block(1), dst.block(0));
      dst.block(1) *= -1.0;
    }

    /*
     *        / I      0    \
     *  dst = |             | * dst
     *        \ 0   -S^{-1} /
     */
    {
      // Copy data from dst.block(1) to vec_tmp_pressure before
      // applying the preconditioner for the pressure block.
      vec_tmp_pressure_ = dst.block(1);
      // Apply preconditioner for pressure/Schur-complement block
      apply_preconditioner_pressure_block(dst.block(1), vec_tmp_pressure_);
    }

    /*
     *        / I  - A^{-1} B^{T} \
     *  dst = |                   | * dst
     *        \ 0          I      /
     */
    {
      // vec_tmp_velocity = B^{T} * dst(1)
      gradient_operator_.apply(vec_tmp_velocity_, dst.block(1));

      // vec_tmp_velocity_2 = A^{-1} * vec_tmp_velocity
      apply_preconditioner_velocity_block(vec_tmp_velocity_2_, vec_tmp_velocity_);

      // dst(0) = dst(0) - vec_tmp_velocity_2
      dst.block(0).add(-1.0, vec_tmp_velocity_2_);
    }
  }
  else
  {
    AssertThrow(false, dealii::ExcNotImplemented());
  }
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::initialize_block_preconditioner()
{
  block_preconditioner_.initialize(this);

  initialize_temporary_vectors();

  initialize_preconditioner_velocity_block();

  initialize_preconditioner_pressure_block();
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::initialize_temporary_vectors()
{
  auto type = this->param_.preconditioner_coupled;

  if(type == IncNS::PreconditionerCoupled::BlockTriangular)
  {
    this->initialize_vector_velocity(vec_tmp_velocity_);
  }
  else if(type == IncNS::PreconditionerCoupled::BlockTriangularFactorization)
  {
    this->initialize_vector_pressure(vec_tmp_pressure_);
    this->initialize_vector_velocity(vec_tmp_velocity_);
    this->initialize_vector_velocity(vec_tmp_velocity_2_);
  }
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::initialize_preconditioner_velocity_block()
{
  auto const type = param_.preconditioner_velocity_block;

  if(type == IncNS::MomentumPreconditioner::InverseMassMatrix)
  {
    preconditioner_velocity_block_ =
      std::make_shared<InverseMassPreconditioner<dim, dim, Number>>(get_matrix_free(),
                                                                    get_dof_index_velocity(),
                                                                    get_quad_index_velocity());
  }
  else
    AssertThrow(type == IncNS::MomentumPreconditioner::None, dealii::ExcNotImplemented());
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::apply_preconditioner_velocity_block(
  OperatorCoupled::VectorType &       dst,
  OperatorCoupled::VectorType const & src) const
{
  auto const type = param_.preconditioner_velocity_block;

  if(type == IncNS::MomentumPreconditioner::InverseMassMatrix)
    preconditioner_velocity_block_->vmult(dst, src);
  else if(type == IncNS::MomentumPreconditioner::None)
    dst = src;
  else
    AssertThrow(false, dealii::ExcNotImplemented());
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::initialize_preconditioner_pressure_block()
{
  auto const type = param_.preconditioner_pressure_block;

  if(type == IncNS::SchurComplementPreconditioner::InverseMassMatrix)
    preconditioner_pressure_block_ =
      std::make_shared<InverseMassPreconditioner<dim, 1, Number>>(get_matrix_free(),
                                                                  get_dof_index_pressure(),
                                                                  get_quad_index_pressure());
  else if(type == IncNS::SchurComplementPreconditioner::LaplaceOperator)
    setup_multigrid_preconditioner_pressure_block();
  else
    AssertThrow(type == IncNS::SchurComplementPreconditioner::None, dealii::ExcNotImplemented());
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::setup_multigrid_preconditioner_pressure_block()
{
  // multigrid V-cycle for negative Laplace operator
  Poisson::LaplaceOperatorData<0, dim> laplace_operator_data;
  laplace_operator_data.dof_index            = get_dof_index_pressure();
  laplace_operator_data.quad_index           = get_quad_index_pressure();
  laplace_operator_data.operator_is_singular = false;
  laplace_operator_data.bc                   = boundary_descriptor_laplace_;

  preconditioner_pressure_block_ = std::make_shared<MultigridPoisson>(mpi_comm_);

  std::shared_ptr<MultigridPoisson> mg_preconditioner =
    std::dynamic_pointer_cast<MultigridPoisson>(preconditioner_pressure_block_);

  mg_preconditioner->initialize(param_.multigrid_data_pressure_block,
                                &get_dof_handler_p().get_triangulation(),
                                grid_->get_coarse_triangulations(),
                                get_dof_handler_p().get_fe(),
                                get_mapping(),
                                laplace_operator_data,
                                false,
                                laplace_operator_data.bc->dirichlet_bc,
                                grid_->periodic_faces);
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::initialize_boundary_descriptor_laplace()
{
  boundary_descriptor_laplace_ = std::make_shared<Poisson::BoundaryDescriptor<0, dim>>();

  // Dirichlet BCs for pressure
  boundary_descriptor_laplace_->dirichlet_bc = boundary_descriptor_->pressure->dirichlet_bc;

  // Neumann BCs for pressure: These boundary conditions are empty, fill with zero functions
  std::for_each(boundary_descriptor_->pressure->neumann_bc.begin(),
                boundary_descriptor_->pressure->neumann_bc.end(),
                [this](auto boundary_id) {
                  this->boundary_descriptor_laplace_->neumann_bc.insert(
                    std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>(
                      boundary_id, std::make_shared<dealii::Functions::ZeroFunction<dim>>(1)));
                });
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::apply_preconditioner_pressure_block(
  OperatorCoupled::VectorType &       dst,
  OperatorCoupled::VectorType const & src) const
{
  auto const type = param_.preconditioner_pressure_block;

  if(type == IncNS::SchurComplementPreconditioner::InverseMassMatrix ||
     type == IncNS::SchurComplementPreconditioner::LaplaceOperator)
    preconditioner_pressure_block_->vmult(dst, src);
  else if(type == IncNS::SchurComplementPreconditioner::None)
    dst = src;
  else
    AssertThrow(false, dealii::ExcNotImplemented())
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::initialize_operators()
{
  dealii::AffineConstraints<Number> constraint_dummy;
  constraint_dummy.close();

  // Mass operator
  {
    MassOperatorData<dim> mass_operator_data;

    mass_operator_data.dof_index  = get_dof_index_velocity();
    mass_operator_data.quad_index = get_quad_index_velocity();

    mass_operator_.initialize(*matrix_free_, constraint_u_, mass_operator_data);
  }

  // Body force operator
  {
    IncNS::RHSOperatorData<dim> data;

    data.dof_index     = get_dof_index_velocity();
    data.quad_index    = get_quad_index_velocity();
    data.kernel_data.f = field_functions_->right_hand_side;

    rhs_operator_.initialize(*matrix_free_, data);
  }

  // Gradient operator
  {
    IncNS::GradientOperatorData<dim> data;

    data.dof_index_velocity   = get_dof_index_velocity();
    data.dof_index_pressure   = get_dof_index_pressure();
    data.quad_index           = get_quad_index_velocity();
    data.integration_by_parts = true;
    data.formulation          = IncNS::FormulationPressureGradientTerm::Weak;
    data.use_boundary_data    = true;
    data.bc                   = boundary_descriptor_->pressure;

    gradient_operator_.initialize(*matrix_free_, data);
  }

  // Divergence operator
  {
    IncNS::DivergenceOperatorData<dim> data;

    data.dof_index_velocity   = get_dof_index_velocity();
    data.dof_index_pressure   = get_dof_index_pressure();
    data.quad_index           = get_quad_index_velocity();
    data.integration_by_parts = true;
    data.formulation          = IncNS::FormulationVelocityDivergenceTerm::Weak;
    data.use_boundary_data    = true;
    data.bc                   = boundary_descriptor_->velocity;

    divergence_operator_.initialize(*matrix_free_, data);
  }

  // Momentum operator
  {
    IncNS::MomentumOperatorData<dim> data;

    data.unsteady_problem   = true;
    data.convective_problem = false;
    data.viscous_problem    = false;
    data.bc                 = boundary_descriptor_->velocity;
    data.dof_index          = get_dof_index_velocity();
    data.quad_index         = get_quad_index_velocity();

    data.use_cell_based_loops = param_.use_cell_based_face_loops;
    data.implement_block_diagonal_preconditioner_matrix_free =
      param_.implement_block_diagonal_preconditioner_matrix_free;
    data.solver_block_diagonal         = Elementwise::Solver::CG;
    data.preconditioner_block_diagonal = Elementwise::Preconditioner::None;
    data.solver_data_block_diagonal    = param_.solver_data_block_diagonal;

    momentum_operator_.initialize(*matrix_free_, constraint_dummy, data);
  }
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::distribute_dofs()
{
  if(param_.spatial_discretization == IncNS::SpatialDiscretization::L2)
  {
    fe_u_ = std::make_shared<dealii::FESystem<dim>>(dealii::FE_DGQ<dim>(param_.degree_u), dim);
  }
  else if(param_.spatial_discretization == IncNS::SpatialDiscretization::HDIV)
  {
    fe_u_ = std::make_shared<dealii::FE_RaviartThomasNodal<dim>>(param_.degree_u - 1);
  }
  else
    AssertThrow(false, dealii::ExcMessage("FE not implemented."));

  // Enumerate DoFs
  dof_handler_u_.distribute_dofs(*fe_u_);
  dof_handler_p_.distribute_dofs(fe_p_);

  unsigned int ndofs_per_cell_velocity;
  if(param_.spatial_discretization == IncNS::SpatialDiscretization::L2)
    ndofs_per_cell_velocity = dealii::Utilities::pow(param_.degree_u + 1, dim) * dim;
  else if(param_.spatial_discretization == IncNS::SpatialDiscretization::HDIV)
    ndofs_per_cell_velocity =
      dealii::Utilities::pow(param_.degree_u, dim - 1) * (param_.degree_u + 1) * dim;
  else
    AssertThrow(false, dealii::ExcMessage("FE not implemented."));

  unsigned int const ndofs_per_cell_pressure =
    dealii::Utilities::pow(param_.get_degree_p(param_.degree_u) + 1, dim);

  pcout_ << "Velocity:" << std::endl;
  if(param_.spatial_discretization == IncNS::SpatialDiscretization::L2)
    print_parameter(pcout_, "degree of 1D polynomials", param_.degree_u);
  else if(param_.spatial_discretization == IncNS::SpatialDiscretization::HDIV)
  {
    print_parameter(pcout_, "degree of 1D polynomials (normal)", param_.degree_u);
    print_parameter(pcout_, "degree of 1D polynomials (tangential)", (param_.degree_u - 1));
  }
  else
    AssertThrow(false, dealii::ExcMessage("FE not implemented."));

  print_parameter(pcout_, "number of dofs per cell", ndofs_per_cell_velocity);
  print_parameter(pcout_, "number of dofs (total)", dof_handler_u_.n_dofs());

  pcout_ << "Pressure:" << std::endl;
  print_parameter(pcout_, "degree of 1D polynomials", param_.get_degree_p(param_.degree_u));
  print_parameter(pcout_, "number of dofs per cell", ndofs_per_cell_pressure);
  print_parameter(pcout_, "number of dofs (total)", dof_handler_p_.n_dofs());

  pcout_ << "Velocity and pressure:" << std::endl;
  print_parameter(pcout_,
                  "number of dofs per cell",
                  ndofs_per_cell_velocity + ndofs_per_cell_pressure);
  print_parameter(pcout_,
                  "number of dofs (total)",
                  dof_handler_u_.n_dofs() + dof_handler_p_.n_dofs());

  pcout_ << std::flush;
}

template class OperatorCoupled<2, float>;
template class OperatorCoupled<2, double>;

template class OperatorCoupled<3, float>;
template class OperatorCoupled<3, double>;

} // namespace Darcy
} // namespace ExaDG