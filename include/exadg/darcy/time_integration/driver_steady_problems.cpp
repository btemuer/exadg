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
#include <deal.II/base/conditional_ostream.h>

// ExaDG
#include <exadg/darcy/postprocessor/postprocessor_interface.h>
#include <exadg/darcy/spatial_discretization/operator_coupled.h>
#include <exadg/darcy/time_integration/driver_steady_problems.h>
#include <exadg/incompressible_navier_stokes/user_interface/parameters.h>
#include <exadg/utilities/print_solver_results.h>

namespace ExaDG
{
namespace Darcy
{
template<int dim, typename Number>
DriverSteadyProblems<dim, Number>::DriverSteadyProblems(
  std::shared_ptr<Operator>                       operator_,
  const IncNS::Parameters &                       param_,
  const MPI_Comm &                                mpi_comm_,
  std::shared_ptr<PostProcessorInterface<Number>> postprocessor_)
  : pde_operator(operator_),
    param(param_),
    mpi_comm(mpi_comm_),
    timer_tree(std::make_shared<TimerTree>()),
    pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_comm_) == 0),
    postprocessor(postprocessor_),
    iterations({0, {0, 0}})
{
}

template<int dim, typename Number>
void
DriverSteadyProblems<dim, Number>::setup()
{
  // initialize global solution vectors (allocation)
  initialize_vectors();

  // initialize solution by using a guess
  initialize_solution();
}

template<int dim, typename Number>
dealii::LinearAlgebra::distributed::Vector<Number> const &
DriverSteadyProblems<dim, Number>::get_velocity() const
{
  return solution.block(0);
}

template<int dim, typename Number>
std::shared_ptr<TimerTree>
DriverSteadyProblems<dim, Number>::get_timings() const
{
  return timer_tree;
}

template<int dim, typename Number>
void
DriverSteadyProblems<dim, Number>::initialize_vectors()
{
  // Solution
  pde_operator->initialize_block_vector_velocity_pressure(solution);

  // Right-hand side
  pde_operator->initialize_block_vector_velocity_pressure(rhs_vector);
}

template<int dim, typename Number>
void
DriverSteadyProblems<dim, Number>::initialize_solution()
{
  pde_operator->prescribe_initial_conditions(solution.block(0), solution.block(1));
}

template<int dim, typename Number>
void
DriverSteadyProblems<dim, Number>::solve()
{
  dealii::Timer timer;
  timer.restart();

  postprocessing();

  do_solve();

  postprocessing();

  timer_tree->insert({"DriverSteady"}, timer.wall_time());
}

template<int dim, typename Number>
void
DriverSteadyProblems<dim, Number>::do_solve()
{
  if(iterations.first == 0)
    global_timer.restart();

  dealii::Timer timer;
  timer.restart();

  pcout << std::endl << "Solve steady state problem:" << std::endl;

  // calculate rhs vector
  pde_operator->rhs(rhs_vector, 0.0 /* time */);

  // solve coupled system of equations
  unsigned int const n_iter =
    pde_operator->solve(solution, rhs_vector, this->param.update_preconditioner_coupled);

  print_solver_info_linear(pcout, n_iter, timer.wall_time());

  iterations.first += 1;
  std::get<1>(iterations.second) += n_iter;

  timer_tree->insert({"DriverSteady", "Solve"}, timer.wall_time());
}

template<int dim, typename Number>
void
DriverSteadyProblems<dim, Number>::postprocessing() const
{
  dealii::Timer timer;
  timer.restart();

  postprocessor->do_postprocessing(solution.block(0), solution.block(1));

  timer_tree->insert({"DriverSteady", "Postprocessing"}, timer.wall_time());
}

template class DriverSteadyProblems<2, float>;
template class DriverSteadyProblems<2, double>;
template class DriverSteadyProblems<3, float>;
template class DriverSteadyProblems<3, double>;
} // namespace Darcy
} // namespace ExaDG