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

#ifndef INCLUDE_EXADG_DARCY_USER_INTERFACE_INPUT_PARAMETERS_H_
#define INCLUDE_EXADG_DARCY_USER_INTERFACE_INPUT_PARAMETERS_H_

// deal.II
#include <deal.II/base/conditional_ostream.h>

// ExaDG
#include <exadg/darcy/user_interface/enum_types.h>
#include <exadg/grid/enum_types.h>
#include <exadg/grid/grid_data.h>
#include <exadg/solvers_and_preconditioners/multigrid/multigrid_parameters.h>
#include <exadg/solvers_and_preconditioners/preconditioners/enum_types.h>
#include <exadg/solvers_and_preconditioners/solvers/solver_data.h>
#include <exadg/time_integration/enum_types.h>
#include <exadg/time_integration/restart_data.h>
#include <exadg/time_integration/solver_info_data.h>

namespace ExaDG
{
namespace Darcy
{
struct MathematicalModelParameters
{
  ProblemType problem_type{ProblemType::Undefined};

  bool right_hand_side{false};

  bool ale{false};

  MeshMovementType mesh_movement_type;
};

struct PhysicalQuantities
{
  double start_time{0.0};
  double end_time{-1.0};
  double viscosity{-1.0};
  double density{-1.0};
};

struct TemporalDiscretizationParameters
{
  TemporalSolverType solver_type{TemporalSolverType::Undefined};

  TemporalDiscretizationMethod method{TemporalDiscretizationMethod::Undefined};

  TimeStepCalculation calculation_of_time_step_size{TimeStepCalculation::UserSpecified};

  double time_step_size{-1.0};

  unsigned int max_number_of_time_steps{std::numeric_limits<unsigned int>::max()};

  unsigned int order_time_integrator{1};

  bool start_with_low_order{true};

  // show solver performance (wall time, number of iterations) every ... timesteps
  SolverInfoData solver_info_data;
};

struct SpatialDiscretizationParameters
{
  GridData grid_data{};

  SpatialDiscretizationMethod method{SpatialDiscretizationMethod::L2};

  unsigned int degree_u{2};

  DegreePressure degree_p{DegreePressure::MixedOrder};

  AdjustPressureLevel adjust_pressure_level{AdjustPressureLevel::ApplyZeroMeanValue};
};

struct BlockJacobiPreconditionerParameters
{
  bool implement_matrix_free{true};

  SolverData solver_data{1000, 1.e-12, 1.e-2, 1000};
};

struct LaplaceOperatorPreconditionerParameters
{
  MultigridData multigrid_data{};

  bool exact_inversion{false};

  SolverData solver_data{10000, 1.e-12, 1.e-6, 100};
};

struct PreconditionerVelocityBlockParameters
{
  VelocityBlockPreconditioner type{VelocityBlockPreconditioner::None};

  BlockJacobiPreconditionerParameters block_jacobi{};
};

struct PreconditionerSchurComplementParameters
{
  SchurComplementPreconditioner type{SchurComplementPreconditioner::None};

  LaplaceOperatorPreconditionerParameters laplace_operator{};
};

struct CoupledPreconditionerParameters
{
  PreconditionerCoupled type{PreconditionerCoupled::None};

  bool update{false};

  unsigned int update_every_time_steps{1};

  PreconditionerVelocityBlockParameters   velocity_block{};
  PreconditionerSchurComplementParameters schur_complement{};
};

struct CoupledDarcySolver
{
  LinearSolverMethod method{LinearSolverMethod::GMRES};

  SolverData data{10000, 1.e-12, 1.e-6, 100};

  CoupledPreconditionerParameters preconditioner{};
};

struct NumericalParameters
{
  bool use_cell_based_face_loops{true};
};

class Parameters
{
public:
  void
  check() const;

  unsigned int
  get_degree_p(unsigned int degree_u) const;

  void
  print(dealii::ConditionalOStream const & pcout, std::string const & name) const;

private:
  void
  print_parameters_mathematical_model(dealii::ConditionalOStream const & pcout) const;

  void
  print_parameters_physical_quantities(dealii::ConditionalOStream const & pcout) const;

  void
  print_parameters_temporal_discretization(dealii::ConditionalOStream const & pcout) const;

  void
  print_parameters_spatial_discretization(dealii::ConditionalOStream const & pcout) const;

  void
  print_parameters_coupled_solver(dealii::ConditionalOStream const & pcout) const;

  void
  print_parameters_numerical(dealii::ConditionalOStream const & pcout) const;

public:
  MathematicalModelParameters      math_model{};
  PhysicalQuantities               physical_quantities{};
  TemporalDiscretizationParameters temporal_disc{};
  SpatialDiscretizationParameters  spatial_disc{};
  CoupledDarcySolver               linear_solver{};
  NumericalParameters              numerical{};
};
} // namespace Darcy
} // namespace ExaDG

#endif /* INCLUDE_EXADG_DARCY_USER_INTERFACE_INPUT_PARAMETERS_H_ */
