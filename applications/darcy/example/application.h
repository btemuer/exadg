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

#ifndef APPLICATIONS_DARCY_TEST_CASES_EXAMPLE_H_
#define APPLICATIONS_DARCY_TEST_CASES_EXAMPLE_H_

namespace ExaDG
{
namespace Darcy
{
enum class BoundaryCondition
{
  VelocityInflow,
};

inline void
string_to_enum(BoundaryCondition & enum_type, std::string const string_type)
{
  // clang-format off
  if     (string_type == "VelocityInflow") enum_type = BoundaryCondition::VelocityInflow;
  else AssertThrow(false, dealii::ExcMessage("Unknown operator type. Not implemented."));
  // clang-format on
}

template<int dim>
class AnalyticalSolutionVelocity : public dealii::Function<dim>
{
public:
  AnalyticalSolutionVelocity(double const inflow_velocity)
    : dealii::Function<dim>(dim, 0.0), inflow_velocity(inflow_velocity)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const
  {
    double result = 0.0;

    if(component == 0)
      result = inflow_velocity;

    return result;
  }

private:
  double const inflow_velocity;
};

template<int dim>
class AnalyticalSolutionPressure : public dealii::Function<dim>
{
public:
  AnalyticalSolutionPressure(double const viscosity,
                             double const permeability,
                             double const inflow_velocity,
                             double const L)
    : dealii::Function<dim>(1 /*n_components*/, 0.0),
      viscosity(viscosity),
      permeability(permeability),
      inflow_velocity(inflow_velocity),
      L(L)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const /*component*/) const
  {
    // pressure decreases linearly in flow direction
    double pressure_gradient = -viscosity * inflow_velocity / permeability;

    double const result = (p[0] - L) * pressure_gradient;

    return result;
  }

private:
  double const viscosity, permeability, inflow_velocity, L;
};

template<int dim>
class DirichletBoundaryVelocity : public dealii::Function<dim>
{
public:
  DirichletBoundaryVelocity(double const inflow_velocity)
    : dealii::Function<dim>(dim, 0.0), inflow_velocity(inflow_velocity)
  {
  }

  double
  value(dealii::Point<dim> const & /*p*/, unsigned int const component = 0) const
  {
    return (component == 0) ? inflow_velocity : 0.0;
  }

private:
  double const inflow_velocity;
};

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : ApplicationBase<dim, Number>(input_file, comm)
  {
  }

  void
  add_parameters(dealii::ParameterHandler & prm) final
  {
    ApplicationBase<dim, Number>::add_parameters(prm);

    // clang-format off
    prm.enter_subsection("Application");
      prm.add_parameter("BoundaryConditionType",
                        boundary_condition_string,
                        "Type of boundary condition.",
                        dealii::Patterns::Selection("VelocityInflow|PressureOutflow"));
      prm.add_parameter("ApplySymmetryBC",
                        apply_symmetry_bc,
                        "Apply symmetry boundary condition.",
                        dealii::Patterns::Bool());
    prm.leave_subsection();
    // clang-format on
  }

private:
  void
  parse_parameters() final
  {
    ApplicationBase<dim, Number>::parse_parameters();

    string_to_enum(boundary_condition, boundary_condition_string);
  }
  void
  set_parameters() final
  {
    // DEFINES DUE TO THE SHARED IncNS PARAMS INTERFACE
    this->param.equation_type                 = IncNS::EquationType::NavierStokes;
    this->param.start_time                    = start_time;
    this->param.end_time                      = end_time;
    this->param.temporal_discretization       = IncNS::TemporalDiscretization::BDFCoupledSolution;
    this->param.calculation_of_time_step_size = IncNS::TimeStepCalculation::UserSpecified;
    this->param.time_step_size                = 1.0e-1;
    this->param.IP_formulation_viscous        = IncNS::InteriorPenaltyFormulation::SIPG;
    this->param.treatment_of_convective_term  = IncNS::TreatmentOfConvectiveTerm::Implicit;
    this->param.apply_penalty_terms_in_postprocessing_step = false;
    this->param.use_continuity_penalty                     = false;
    this->param.use_divergence_penalty                     = false;

    // MATHEMATICAL MODEL
    this->param.problem_type    = IncNS::ProblemType::Steady;
    this->param.right_hand_side = false;

    // PHYSICAL QUANTITIES
    this->param.viscosity = viscosity;
    this->param.density   = density;

    // TEMPORAL DISCRETIZATION
    this->param.solver_type = IncNS::SolverType::Steady;

    this->param.convergence_criterion_steady_problem =
      IncNS::ConvergenceCriterionSteadyProblem::SolutionIncrement;
    this->param.abs_tol_steady = 1.e-12;
    this->param.rel_tol_steady = 1.e-6;

    // output of solver information
    // this->param.solver_info_data.interval_time =
    //  (this->param.end_time - this->param.start_time) / 10;

    // SPATIAL DISCRETIZATION
    this->param.spatial_discretization  = IncNS::SpatialDiscretization::L2;
    this->param.grid.triangulation_type = TriangulationType::Distributed;
    this->param.grid.n_refine_global    = 9;
    this->param.degree_u                = 3;
    this->param.grid.mapping_degree     = this->param.degree_u;
    this->param.degree_p                = IncNS::DegreePressure::MixedOrder;

    // COUPLED NAVIER-STOKES SOLVER

    // linear solver
    this->param.solver_coupled      = IncNS::SolverCoupled::FGMRES;
    this->param.solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-6, 200);

    // preconditioning linear solver
    this->param.preconditioner_coupled        = IncNS::PreconditionerCoupled::BlockTriangular;

    // preconditioner velocity/momentum block
    this->param.preconditioner_velocity_block = IncNS::MomentumPreconditioner::InverseMassMatrix;

    // preconditioner Schur-complement block
    this->param.preconditioner_pressure_block = IncNS::SchurComplementPreconditioner::LaplaceOperator;
  }

  void
  create_grid() final
  {
    double const              y_upper = H / 2;
    double const              y_lower = -H / 2;
    dealii::Point<dim>        point1(0.0, y_lower), point2(L, y_upper);
    std::vector<unsigned int> repetitions({1, 1});
    dealii::GridGenerator::subdivided_hyper_rectangle(*this->grid->triangulation,
                                                      repetitions,
                                                      point1,
                                                      point2);

    // set boundary indicator
    for(auto cell : this->grid->triangulation->cell_iterators())
    {
      for(auto const & face : cell->face_indices())
      {
        if((std::fabs(cell->face(face)->center()(0) - 0.0) < 1e-12))
          cell->face(face)->set_boundary_id(1);
        if((std::fabs(cell->face(face)->center()(0) - L) < 1e-12))
          cell->face(face)->set_boundary_id(2);
      }
    }

    this->grid->triangulation->refine_global(this->param.grid.n_refine_global);
  }

  void
  set_boundary_descriptor() final
  {
    using pair =
      typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>;

    AssertThrow(boundary_condition == BoundaryCondition::VelocityInflow,
                dealii::ExcMessage("not implemented."));

    // fill boundary descriptor velocity
    {
      // inflow
      {
        this->boundary_descriptor->velocity->dirichlet_bc.insert(
          pair(1, new DirichletBoundaryVelocity<dim>(this->inflow_velocity)));
      }
      // outflow
      {
        this->boundary_descriptor->velocity->neumann_bc.insert(
          pair(2, new dealii::Functions::ZeroFunction<dim>(dim)));
      }

      // slip-walls (symmetry bc)
      if(apply_symmetry_bc)
      {
        // slip boundary condition: always u*n=0
        // function will not be used -> use ZeroFunction
        this->boundary_descriptor->velocity->symmetry_bc.insert(
          pair(0, new dealii::Functions::ZeroFunction<dim>(dim)));
      }
      else
      {
        // outflow
        this->boundary_descriptor->velocity->neumann_bc.insert(
          pair(0, new dealii::Functions::ZeroFunction<dim>(dim)));
      }
    }
    // fill boundary descriptor pressure
    {
      {
        // inflow
        this->boundary_descriptor->pressure->neumann_bc.insert(1);
        // outflow
        this->boundary_descriptor->pressure->dirichlet_bc.insert(
          pair(2, new dealii::Functions::ConstantFunction<dim>(0.0, 1)));
      }

      if(apply_symmetry_bc)
      {
        // slip-walls
        this->boundary_descriptor->pressure->neumann_bc.insert(0);
      }
      else
      {
        // outflow
        this->boundary_descriptor->pressure->dirichlet_bc.insert(
          pair(0, new dealii::Functions::ConstantFunction<dim>(0.0, 1)));
      }
    }
  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution_velocity.reset(
      new dealii::Functions::ConstantFunction<dim>(0.0, dim));
    this->field_functions->initial_solution_pressure.reset(
      new dealii::Functions::ConstantFunction<dim>(0.0, 1));
    this->field_functions->analytical_solution_pressure.reset(new AnalyticalSolutionPressure<dim>(
      this->viscosity, this->permeability, this->inflow_velocity, this->L));
    this->field_functions->right_hand_side.reset(new dealii::Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<PostProcessor<dim, Number>>
  create_postprocessor() final
  {
    Darcy::PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.time_control_data.is_active = this->output_parameters.write;
    pp_data.output_data.directory                   = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename                    = this->output_parameters.filename;
    pp_data.output_data.write_vorticity             = false;
    pp_data.output_data.write_divergence            = false;
    pp_data.output_data.write_velocity_magnitude    = false;
    pp_data.output_data.write_vorticity_magnitude   = false;
    pp_data.output_data.write_processor_id          = false;
    pp_data.output_data.write_q_criterion           = false;
    pp_data.output_data.degree                      = this->param.degree_u;
    pp_data.output_data.write_higher_order          = true;

    std::shared_ptr<PostProcessor<dim, Number>> pp;
    pp.reset(new Darcy::PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }

  std::string       boundary_condition_string = "VelocityInflow";
  BoundaryCondition boundary_condition        = BoundaryCondition::VelocityInflow;

  bool apply_symmetry_bc = true;

  IncNS::FormulationViscousTerm const formulation_viscous_term =
    IncNS::FormulationViscousTerm::LaplaceFormulation;

  double const inflow_velocity = 1.0e-3;
  double const viscosity       = 1.0e-1;
  double const density         = 1.0e-1;
  double const permeability    = 1.0e-9;

  double const H = 1.0;
  double const L = 1.0;

  double const start_time = 0.0;
  double const end_time   = 100.0;
};

} // namespace Darcy

} // namespace ExaDG

#include <exadg/darcy/user_interface/implement_get_application.h>


#endif // APPLICATIONS_DARCY_TEST_CASES_EXAMPLE_H_
