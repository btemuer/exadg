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
#include <deal.II/base/exceptions.h>

// ExaDG
#include <exadg/darcy/user_interface/parameters.h>
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
namespace Darcy
{
void
Parameters::check() const
{
  // MATHEMATICAL MODEL
  AssertThrow(math_model.problem_type != ProblemType::Undefined,
              dealii::ExcMessage("parameter must be defined"));

  // ALE
  if(math_model.ale)
  {
    AssertThrow(spatial_disc.method == SpatialDiscretizationMethod::L2,
                dealii::ExcMessage(
                  "ALE is currently only implemented for L2 conforming function spaces."));

    AssertThrow(
      math_model.problem_type == ProblemType::Unsteady &&
        temporal_disc.solver_type == TemporalSolverType::Unsteady,
      dealii::ExcMessage(
        "Both problem type and solver type have to be Unsteady when using ALE formulation."));
  }

  // PHYSICAL QUANTITIES
  AssertThrow(physical_quantities.end_time > physical_quantities.start_time,
              dealii::ExcMessage("parameter end_time must be defined (properly)"));
  AssertThrow(physical_quantities.density > 0.0, dealii::ExcMessage("parameter must be defined"));
  AssertThrow(physical_quantities.viscosity > 0.0, dealii::ExcMessage("parameter must be defined"));

  // TEMPORAL DISCRETIZATION
  AssertThrow(temporal_disc.solver_type != TemporalSolverType::Undefined,
              dealii::ExcMessage("parameter must be defined"));

  if(math_model.problem_type == ProblemType::Unsteady)
  {
    AssertThrow(temporal_disc.solver_type == TemporalSolverType::Unsteady,
                dealii::ExcMessage(
                  "An unsteady solver has to be used to solve unsteady problems."));

    AssertThrow(temporal_disc.method != TemporalDiscretizationMethod::Undefined,
                dealii::ExcMessage("parameter must be defined"));

    AssertThrow(temporal_disc.calculation_of_time_step_size != TimeStepCalculation::Undefined,
                dealii::ExcMessage("parameter must be defined"));

    if(temporal_disc.calculation_of_time_step_size == TimeStepCalculation::UserSpecified)
      AssertThrow(temporal_disc.time_step_size > 0.,
                  dealii::ExcMessage("parameter must be defined"));
  }

  // SPATIAL DISCRETIZATION
  spatial_disc.grid_data.check();

  // COUPLED DARCY SOLVER
  if(temporal_disc.method == TemporalDiscretizationMethod::BDFCoupled)
  {
    if(linear_solver.preconditioner.velocity_block.block_jacobi.implement_matrix_free)
    {
      AssertThrow(
        numerical.use_cell_based_face_loops,
        dealii::ExcMessage(
          "Cell based face loops have to be used for matrix-free implementation of block diagonal preconditioner."));
    }
  }
}

unsigned int
Parameters::get_degree_p(unsigned int const degree_u) const
{
  unsigned int k = 1;

  if(spatial_disc.degree_p == DegreePressure::MixedOrder)
  {
    AssertThrow(degree_u > 0,
                dealii::ExcMessage("The polynomial degree of the velocity shape functions"
                                   " has to be larger than zero for a mixed-order formulation."));

    k = degree_u - 1;
  }
  else if(spatial_disc.degree_p == DegreePressure::EqualOrder)
    k = degree_u;
  else
    AssertThrow(false, dealii::ExcNotImplemented());

  return k;
}

void
Parameters::print(dealii::ConditionalOStream const & pcout, std::string const & name) const
{
  pcout << std::endl << name << std::endl;

  // MATHEMATICAL MODEL
  print_parameters_mathematical_model(pcout);

  // PHYSICAL QUANTITIES
  print_parameters_physical_quantities(pcout);

  // TEMPORAL DISCRETIZATION
  if(temporal_disc.solver_type == TemporalSolverType::Unsteady)
    print_parameters_temporal_discretization(pcout);

  // SPATIAL DISCRETIZATION
  print_parameters_spatial_discretization(pcout);

  print_parameters_coupled_solver(pcout);

  // NUMERICAL PARAMETERS
  print_parameters_numerical(pcout);
}

void
Parameters::print_parameters_mathematical_model(dealii::ConditionalOStream const & pcout) const
{
  pcout << std::endl << "Mathematical model:" << std::endl;

  print_parameter(pcout, "Problem type", enum_to_string(math_model.problem_type));

  print_parameter(pcout, "Right-hand side", math_model.right_hand_side);

  print_parameter(pcout, "Use ALE formulation", math_model.ale);
  if(math_model.ale)
  {
    print_parameter(pcout, "Mesh movement type", enum_to_string(math_model.mesh_movement_type));
  }
}

void
Parameters::print_parameters_physical_quantities(dealii::ConditionalOStream const & pcout) const
{
  pcout << std::endl << "Physical quantities:" << std::endl;

  // Start and end time
  if(temporal_disc.solver_type == TemporalSolverType::Unsteady)
  {
    print_parameter(pcout, "Start time", physical_quantities.start_time);
    print_parameter(pcout, "End time", physical_quantities.end_time);
  }

  // Viscosity
  print_parameter(pcout, "Viscosity", physical_quantities.viscosity);

  // Density
  print_parameter(pcout, "Density", physical_quantities.density);
}

void
Parameters::print_parameters_temporal_discretization(dealii::ConditionalOStream const & pcout) const
{
  pcout << std::endl << "Temporal discretization:" << std::endl;

  print_parameter(pcout, "Temporal discretization method", enum_to_string(temporal_disc.method));

  print_parameter(pcout,
                  "Calculation of time step size",
                  enum_to_string(temporal_disc.calculation_of_time_step_size));

  // here we do not print quantities such as max_velocity, cfl, time_step_size
  // because this is done by the time integration scheme (or the functions that
  // calculate the time step size)

  print_parameter(pcout, "Maximum number of time steps", temporal_disc.max_number_of_time_steps);
  print_parameter(pcout, "Order of time integration scheme", temporal_disc.order_time_integrator);
  print_parameter(pcout, "Start with low order method", temporal_disc.start_with_low_order);

  // output of solver information
  temporal_disc.solver_info_data.print(pcout);
}

void
Parameters::print_parameters_spatial_discretization(dealii::ConditionalOStream const & pcout) const
{
  pcout << std::endl << "Spatial discretization:" << std::endl;

  spatial_disc.grid_data.print(pcout);

  print_parameter(pcout, "Element type", enum_to_string(spatial_disc.method));

  if(spatial_disc.method == SpatialDiscretizationMethod::L2)
    print_parameter(pcout, "Polynomial degree velocity", spatial_disc.degree_u);
  else
    AssertThrow(false, dealii::ExcNotImplemented());

  print_parameter(pcout, "Polynomial degree pressure", enum_to_string(spatial_disc.degree_p));

  print_parameter(pcout, "Ale formulation (with poroelasticity coupling)", math_model.ale);

  print_parameter(pcout,
                  "Adjust pressure level (if undefined)",
                  enum_to_string(spatial_disc.adjust_pressure_level));
}

void
Parameters::print_parameters_coupled_solver(dealii::ConditionalOStream const & pcout) const
{
  pcout << std::endl << "Coupled Navier-Stokes solver:" << std::endl;

  if(linear_solver.preconditioner.update)
  {
    print_parameter(pcout,
                    "Update every time steps",
                    linear_solver.preconditioner.update_every_time_steps);
  }

  pcout << std::endl << "  Velocity/momentum block:" << std::endl;

  print_parameter(pcout,
                  "Preconditioner",
                  enum_to_string(linear_solver.preconditioner.velocity_block.type));

  if(linear_solver.preconditioner.velocity_block.block_jacobi.implement_matrix_free)
    linear_solver.preconditioner.velocity_block.block_jacobi.solver_data.print(pcout);

  pcout << std::endl << "  Pressure/Schur-complement block:" << std::endl;

  print_parameter(pcout,
                  "Preconditioner",
                  enum_to_string(linear_solver.preconditioner.schur_complement.type));

  if(linear_solver.preconditioner.schur_complement.type ==
     SchurComplementPreconditioner::LaplaceOperator)
  {
    linear_solver.preconditioner.schur_complement.laplace_operator.multigrid_data.print(pcout);

    print_parameter(pcout,
                    "Exact inversion of Laplace operator",
                    linear_solver.preconditioner.schur_complement.laplace_operator.exact_inversion);

    if(linear_solver.preconditioner.schur_complement.laplace_operator.exact_inversion)
      linear_solver.preconditioner.schur_complement.laplace_operator.solver_data.print(pcout);
  }
}

void
Parameters::print_parameters_numerical(dealii::ConditionalOStream const & pcout) const
{
  pcout << std::endl << "Numerical parameters:" << std::endl;

  print_parameter(pcout, "Use cell-based face loops", numerical.use_cell_based_face_loops);
}
} // namespace Darcy
} // namespace ExaDG