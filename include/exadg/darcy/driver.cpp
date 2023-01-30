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

#include <exadg/darcy/driver.h>
#include <exadg/grid/get_dynamic_mapping.h>
#include <exadg/utilities/print_solver_results.h>

namespace ExaDG
{
namespace Darcy
{
template<int dim, typename Number>
Driver<dim, Number>::Driver(MPI_Comm const &                              comm,
                            std::shared_ptr<ApplicationBase<dim, Number>> application)
  : mpi_comm(comm),
    pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(comm) == 0),
    application(application)
{
  print_general_info<Number>(pcout, mpi_comm, false);
}

template<int dim, typename Number>
void
Driver<dim, Number>::setup()
{
  dealii::Timer timer;
  timer.restart();

  pcout << std::endl << "Setting up Darcy solver: " << std::endl;

  application->setup();

  // moving mesh (ALE formulation)
  if(application->get_parameters().math_model.ale)
  {
    if(application->get_parameters().math_model.mesh_movement_type == MeshMovementType::Function)
    {
      std::shared_ptr<dealii::Function<dim>> mesh_motion =
        application->create_mesh_movement_function();

      grid_motion = std::make_shared<GridMotionFunction<dim, Number>>(
        application->get_grid()->mapping,
        application->get_parameters().spatial_disc.grid_data.mapping_degree,
        *application->get_grid()->triangulation,
        mesh_motion,
        application->get_parameters().physical_quantities.start_time);
    }
    else
      AssertThrow(false, dealii::ExcNotImplemented());
  }

  pde_operator =
    std::make_shared<Darcy::OperatorCoupled<dim, Number>>(application->get_grid(),
                                                          grid_motion,
                                                          application->get_boundary_descriptor(),
                                                          application->get_field_functions(),
                                                          application->get_parameters(),
                                                          "fluid",
                                                          mpi_comm);

  // Initialize matrix-free evaluation
  matrix_free_data = std::make_shared<MatrixFreeData<dim, Number>>();
  matrix_free_data->append(pde_operator);

  matrix_free = std::make_shared<dealii::MatrixFree<dim, Number>>();

  if(application->get_parameters().numerical.use_cell_based_face_loops)
  {
    Categorization::do_cell_based_loops(*application->get_grid()->triangulation,
                                        matrix_free_data->data);
  }

  std::shared_ptr<dealii::Mapping<dim> const> mapping =
    get_dynamic_mapping<dim, Number>(application->get_grid(), grid_motion);

  matrix_free->reinit(*mapping,
                      matrix_free_data->get_dof_handler_vector(),
                      matrix_free_data->get_constraint_vector(),
                      matrix_free_data->get_quadrature_vector(),
                      matrix_free_data->data);

  // setup Darcy operator
  pde_operator->setup(matrix_free, matrix_free_data);

  // setup postprocessor
  postprocessor = application->create_postprocessor();
  postprocessor->setup(*pde_operator);

  // setup time integrator before calling setup_solvers()
  if(application->get_parameters().temporal_disc.solver_type == TemporalSolverType::Unsteady)
  {
    time_integrator = std::make_shared<TimeIntBDFCoupled<dim, Number>>(
      pde_operator, application->get_parameters(), mpi_comm, postprocessor);

    time_integrator->setup(false); // no restart

    pde_operator->setup_solvers(time_integrator->get_scaling_factor_time_derivative_term());
  }
  else if(application->get_parameters().temporal_disc.solver_type == TemporalSolverType::Steady)
  {
    // initialize driver for steady state problem that depends on pde_operator
    driver_steady = std::make_shared<DriverSteadyProblems<dim, Number>>(
      pde_operator, application->get_parameters(), mpi_comm, postprocessor);

    driver_steady->setup();

    pde_operator->setup_solvers(1.0 /* dummy */);
  }
  else
    AssertThrow(false, dealii::ExcNotImplemented());

  timer_tree.insert({"Darcy flow", "Setup"}, timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::ale_update() const
{
  // move the mesh and update dependent data structures
  dealii::Timer timer;
  timer.restart();

  dealii::Timer sub_timer;

  sub_timer.restart();
  grid_motion->update(time_integrator->get_next_time(), false);
  timer_tree.insert({"Darcy flow", "ALE", "Reinit mapping"}, sub_timer.wall_time());

  sub_timer.restart();
  std::shared_ptr<dealii::Mapping<dim> const> mapping =
    get_dynamic_mapping<dim, Number>(application->get_grid(), grid_motion);
  matrix_free->update_mapping(*mapping);
  timer_tree.insert({"Darcy flow", "ALE", "Update matrix-free"}, sub_timer.wall_time());

  sub_timer.restart();
  pde_operator->update_after_grid_motion();
  timer_tree.insert({"Darcy flow", "ALE", "Update operator"}, sub_timer.wall_time());

  sub_timer.restart();
  time_integrator->ale_update();
  timer_tree.insert({"Darcy flow", "ALE", "Update time integrator"}, sub_timer.wall_time());

  timer_tree.insert({"Darcy flow", "ALE"}, timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::solve() const
{
  if(application->get_parameters().math_model.problem_type == ProblemType::Unsteady)
  {
    if(application->get_parameters().math_model.ale == true)
    {
      while(not time_integrator->finished())
      {
        time_integrator->advance_one_timestep_pre_solve(true);

        ale_update();

        time_integrator->advance_one_timestep_solve();

        time_integrator->advance_one_timestep_post_solve();
      }
    }
    else
      time_integrator->timeloop();
  }
  else if(application->get_parameters().math_model.problem_type == ProblemType::Steady)
  {
    if(application->get_parameters().temporal_disc.solver_type == TemporalSolverType::Steady)
      driver_steady->solve();
    else
      AssertThrow(false, dealii::ExcNotImplemented());
  }
  else
    AssertThrow(false, dealii::ExcNotImplemented());
}

template<int dim, typename Number>
void
Driver<dim, Number>::print_performance_results(double const total_time) const
{
  pcout << std::endl
        << "_________________________________________________________________________________"
        << std::endl
        << std::endl;

  pcout << "Performance results for Darcy solver:" << std::endl;

  // Iterations
  {
    if(application->get_parameters().temporal_disc.solver_type == TemporalSolverType::Unsteady)
    {
      pcout << std::endl << "Average number of iterations:" << std::endl;

      time_integrator->print_iterations();
    }
  }

  // Wall times
  {
    timer_tree.insert({"Darcy flow"}, total_time);

    if(application->get_parameters().temporal_disc.solver_type == TemporalSolverType::Unsteady)
    {
      timer_tree.insert({"Darcy flow"}, time_integrator->get_timings());
    }
    else if(application->get_parameters().temporal_disc.solver_type == TemporalSolverType::Steady)
    {
      timer_tree.insert({"Darcy flow"}, driver_steady->get_timings());
    }
    else
      AssertThrow(false, dealii::ExcNotImplemented());
  }

  // Performance
  {
    dealii::types::global_dof_index const DoFs = pde_operator->get_number_of_dofs();
    unsigned int const N_mpi_processes         = dealii::Utilities::MPI::n_mpi_processes(mpi_comm);

    dealii::Utilities::MPI::MinMaxAvg overall_time_data =
      dealii::Utilities::MPI::min_max_avg(total_time, mpi_comm);
    double const overall_time_avg = overall_time_data.avg;

    // Throughput in DoFs/s per time step per core
    {
      if(application->get_parameters().temporal_disc.solver_type == TemporalSolverType::Unsteady)
      {
        unsigned int const N_time_steps = time_integrator->get_number_of_time_steps();
        print_throughput_unsteady(pcout, DoFs, overall_time_avg, N_time_steps, N_mpi_processes);
      }
      else if(application->get_parameters().temporal_disc.solver_type == TemporalSolverType::Steady)
      {
        print_throughput_steady(pcout, DoFs, overall_time_avg, N_mpi_processes);
      }
      else
        AssertThrow(false, dealii::ExcNotImplemented());
    }

    // Computational costs in CPUh
    print_costs(pcout, overall_time_avg, N_mpi_processes);
  }

  pcout << "_________________________________________________________________________________"
        << std::endl
        << std::endl;
}

template class Driver<2, float>;
template class Driver<2, double>;

template class Driver<3, float>;
template class Driver<3, double>;

} // namespace Darcy
} // namespace ExaDG