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
#include <exadg/utilities/print_solver_results.h>

namespace ExaDG
{
namespace Darcy
{
template<int dim, typename Number>
Driver<dim, Number>::Driver(MPI_Comm const &                              comm,
                            std::shared_ptr<ApplicationBase<dim, Number>> app)
  : mpi_comm(comm),
    pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(comm) == 0),
    application(app)
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

  // no mesh movement -> no ALE setup

  pde_operator =
    std::make_shared<Darcy::OperatorCoupled<dim, Number>>(application->get_grid(),
                                                          application->get_boundary_descriptor(),
                                                          application->get_field_functions(),
                                                          application->get_parameters(),
                                                          "fluid",
                                                          mpi_comm);


  // Initialize matrix-free evaluation
  matrix_free_data = std::make_shared<MatrixFreeData<dim, Number>>();
  matrix_free_data->append(pde_operator);

  matrix_free = std::make_shared<dealii::MatrixFree<dim, Number>>();

  if(application->get_parameters().use_cell_based_face_loops)
  {
    Categorization::do_cell_based_loops(*application->get_grid()->triangulation,
                                        matrix_free_data->data);
  }

  // no dynamic mapping - (no moving mesh, no ALE)
  matrix_free->reinit(*(application->get_grid()->mapping),
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
  if(application->get_parameters().solver_type == IncNS::SolverType::Unsteady)
  {
    time_integrator = std::make_shared<TimeIntBDFCoupled<dim, Number>>(
      pde_operator, application->get_parameters(), mpi_comm, postprocessor);

    time_integrator->setup(application->get_parameters().restarted_simulation);

    pde_operator->setup_solvers(time_integrator->get_scaling_factor_time_derivative_term());
  }
  else if(application->get_parameters().solver_type == IncNS::SolverType::Steady)
  {
    auto operator_coupled = std::dynamic_pointer_cast<OperatorCoupled<dim, Number>>(pde_operator);

    // initialize driver for steady state problem that depends on pde_operator
    driver_steady = std::make_shared<DriverSteadyProblems<dim, Number>>(
      operator_coupled, application->get_parameters(), mpi_comm, postprocessor);

    driver_steady->setup();

    pde_operator->setup_solvers(1.0 /* dummy */);
  }
  else
    AssertThrow(false, dealii::ExcMessage("Not implemented."));

  timer_tree.insert({"Darcy flow", "Setup"}, timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::solve() const
{
  if (application->get_parameters().problem_type == IncNS::ProblemType::Unsteady)
  {
    time_integrator->timeloop();
  }
  else if(application->get_parameters().problem_type == IncNS::ProblemType::Steady)
  {
    if(application->get_parameters().solver_type == IncNS::SolverType::Steady)
      driver_steady->solve();
    else
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
  }
  else
    AssertThrow(false, dealii::ExcMessage("Not implemented."));
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
    if(application->get_parameters().solver_type == IncNS::SolverType::Unsteady)
    {
      pcout << std::endl << "Average number of iterations:" << std::endl;

      time_integrator->print_iterations();
    }
  }

  // Wall times
  {
    timer_tree.insert({"Darcy flow"}, total_time);

    if(application->get_parameters().solver_type == IncNS::SolverType::Unsteady)
    {
      timer_tree.insert({"Darcy flow"}, time_integrator->get_timings());
    }
    else if(application->get_parameters().solver_type == IncNS::SolverType::Steady)
    {
      timer_tree.insert({"Darcy flow"}, driver_steady->get_timings());
    }
    else
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
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
      if(application->get_parameters().solver_type == IncNS::SolverType::Unsteady)
      {
        unsigned int const N_time_steps = time_integrator->get_number_of_time_steps();
        print_throughput_unsteady(pcout, DoFs, overall_time_avg, N_time_steps, N_mpi_processes);
      }
      else if(application->get_parameters().solver_type == IncNS::SolverType::Steady)
      {
        print_throughput_steady(pcout, DoFs, overall_time_avg, N_mpi_processes);
      }
      else
        AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }

    // Computational costs in CPUh
    print_costs(pcout, overall_time_avg, N_mpi_processes);
  }

  pcout << "_________________________________________________________________________________"
        << std::endl
        << std::endl;
}

template class Driver<2, float>;
template class Driver<3, float>;
template class Driver<2, double>;
template class Driver<3, double>;

} // namespace Darcy
} // namespace ExaDG