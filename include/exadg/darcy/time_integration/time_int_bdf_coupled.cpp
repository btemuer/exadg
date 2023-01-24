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

#include <exadg/darcy/postprocessor/postprocessor_interface.h>
#include <exadg/darcy/spatial_discretization/operator_coupled.h>
#include <exadg/darcy/time_integration/time_int_bdf_coupled.h>
#include <exadg/incompressible_navier_stokes/user_interface/parameters.h>
#include <exadg/time_integration/push_back_vectors.h>
#include <exadg/time_integration/time_step_calculation.h>
#include <exadg/utilities/print_solver_results.h>

namespace ExaDG
{
namespace Darcy
{
template<int dim, typename Number>
TimeIntBDFCoupled<dim, Number>::TimeIntBDFCoupled(
  std::shared_ptr<Operator>                       pde_operator,
  Parameters const &                              param,
  MPI_Comm const &                                mpi_comm,
  std::shared_ptr<PostProcessorInterface<Number>> postprocessor)
  : TimeIntBDFBase<Number>(param.physical_quantities.start_time,
                           param.physical_quantities.end_time,
                           param.temporal_disc.max_number_of_time_steps,
                           param.temporal_disc.order_time_integrator,
                           param.temporal_disc.start_with_low_order,
                           false, // no adaptive time stepping
                           {},    // no restart
                           mpi_comm,
                           false),
    param(param),
    use_extrapolation(true),
    pde_operator(pde_operator),
    solution(this->order),
    iterations({0, 0}),
    postprocessor(postprocessor)
{
}

template<int dim, typename Number>
void
TimeIntBDFCoupled<dim, Number>::allocate_vectors()
{
  // solution
  for(unsigned int i = 0; i < solution.size(); ++i)
    pde_operator->initialize_block_vector_velocity_pressure(solution[i]);
  pde_operator->initialize_block_vector_velocity_pressure(solution_np);
}

template<int dim, typename Number>
void
TimeIntBDFCoupled<dim, Number>::setup_derived()
{
  // nothing
}

template<int dim, typename Number>
void
TimeIntBDFCoupled<dim, Number>::read_restart_vectors(boost::archive::binary_iarchive & ia)
{
  for(unsigned int i = 0; i < this->order; i++)
  {
    VectorType tmp = get_velocity(i);
    ia >> tmp;
    set_velocity(tmp, i);
  }
  for(unsigned int i = 0; i < this->order; i++)
  {
    VectorType tmp = get_pressure(i);
    ia >> tmp;
    set_pressure(tmp, i);
  }
}

template<int dim, typename Number>
void
TimeIntBDFCoupled<dim, Number>::write_restart_vectors(boost::archive::binary_oarchive & oa) const
{
  for(unsigned int i = 0; i < this->order; i++)
  {
    oa << get_velocity(i);
  }
  for(unsigned int i = 0; i < this->order; i++)
  {
    oa << get_pressure(i);
  }
}

template<int dim, typename Number>
void
TimeIntBDFCoupled<dim, Number>::initialize_current_solution()
{
  pde_operator->prescribe_initial_conditions(solution[0].block(0),
                                             solution[0].block(1),
                                             this->get_time());
}

template<int dim, typename Number>
void
TimeIntBDFCoupled<dim, Number>::initialize_former_solutions()
{
  // note that the loop begins with i=1! (we could also start with i=0 but this is not necessary) ??
  for(unsigned int i = 1; i < solution.size(); ++i)
  {
    pde_operator->prescribe_initial_conditions(solution[i].block(0),
                                               solution[i].block(1),
                                               this->get_previous_time(i));
  }
}

template<int dim, typename Number>
double
TimeIntBDFCoupled<dim, Number>::calculate_time_step_size()
{
  double time_step{};

  if(param.temporal_disc.calculation_of_time_step_size ==
     TimeStepCalculation::UserSpecified)
  {
    time_step =
      calculate_const_time_step(param.temporal_disc.time_step_size, 0 /* no refinement*/);

    this->pcout << std::endl << "User specified time step size:" << std::endl << std::endl;
    print_parameter(this->pcout, "time step size", time_step);
  }
  else
    AssertThrow(false,
                dealii::ExcMessage("Specified type of time step calculation is not implemented."));

  return time_step;
}

template<int dim, typename Number>
double
TimeIntBDFCoupled<dim, Number>::recalculate_time_step_size() const
{
  AssertThrow(false, dealii::ExcMessage("Adaptive time step is not implemented."));
}

template<int dim, typename Number>
typename TimeIntBDFCoupled<dim, Number>::VectorType const &
TimeIntBDFCoupled<dim, Number>::get_velocity() const
{
  return solution[0].block(0);
}

template<int dim, typename Number>
typename TimeIntBDFCoupled<dim, Number>::VectorType const &
TimeIntBDFCoupled<dim, Number>::get_velocity(unsigned int i) const
{
  return solution[i].block(0);
}

template<int dim, typename Number>
typename TimeIntBDFCoupled<dim, Number>::VectorType const &
TimeIntBDFCoupled<dim, Number>::get_velocity_np() const
{
  return solution_np.block(0);
}

template<int dim, typename Number>
typename TimeIntBDFCoupled<dim, Number>::VectorType const &
TimeIntBDFCoupled<dim, Number>::get_pressure() const
{
  return solution[0].block(1);
}

template<int dim, typename Number>
typename TimeIntBDFCoupled<dim, Number>::VectorType const &
TimeIntBDFCoupled<dim, Number>::get_pressure_np() const
{
  return solution_np.block(1);
}

template<int dim, typename Number>
typename TimeIntBDFCoupled<dim, Number>::VectorType const &
TimeIntBDFCoupled<dim, Number>::get_pressure(unsigned int i) const
{
  return solution[i].block(1);
}

template<int dim, typename Number>
void
TimeIntBDFCoupled<dim, Number>::set_velocity(VectorType const & velocity_in, unsigned int const i)
{
  solution[i].block(0) = velocity_in;
}

template<int dim, typename Number>
void
TimeIntBDFCoupled<dim, Number>::set_pressure(VectorType const & pressure_in, unsigned int const i)
{
  solution[i].block(1) = pressure_in;
}

template<int dim, typename Number>
void
TimeIntBDFCoupled<dim, Number>::get_velocities_and_times(
  std::vector<VectorType const *> & velocities,
  std::vector<double> &             times) const
{
  /*
   *   time t
   *  -------->   t_{n-2}   t_{n-1}   t_{n}     t_{n+1}
   *  _______________|_________|________|___________|___________\
   *                 |         |        |           |           /
   *               sol[2]    sol[1]   sol[0]
   *             times[2]  times[1]  times[0]
   */
  unsigned int current_order = this->order;
  if(this->time_step_number <= this->order &&
     this->param.temporal_disc.start_with_low_order == true)
  {
    current_order = this->time_step_number;
  }

  AssertThrow(current_order > 0 && current_order <= this->order,
              dealii::ExcMessage("Invalid parameter current_order"));

  velocities.resize(current_order);
  times.resize(current_order);

  for(unsigned int i = 0; i < current_order; ++i)
  {
    velocities.at(i) = &get_velocity(i);
    times.at(i)      = this->get_previous_time(i);
  }
}

template<int dim, typename Number>
void
TimeIntBDFCoupled<dim, Number>::get_velocities_and_times_np(
  std::vector<VectorType const *> & velocities,
  std::vector<double> &             times) const
{
  /*
   *   time t
   *  -------->     t_{n-2}   t_{n-1}   t_{n}     t_{n+1}
   *  _______________|_________|________|___________|___________\
   *                 |         |        |           |           /
   *               sol[3]   sol[2]    sol[1]     sol[0]
   *              times[3] times[2]  times[1]   times[0]
   */
  unsigned int current_order = this->order;
  if(this->time_step_number <= this->order &&
     this->param.temporal_disc.start_with_low_order == true)
  {
    current_order = this->time_step_number;
  }

  AssertThrow(current_order > 0 && current_order <= this->order,
              dealii::ExcMessage("Invalid parameter current_order"));

  velocities.resize(current_order + 1);
  times.resize(current_order + 1);

  velocities.at(0) = &get_velocity_np();
  times.at(0)      = this->get_next_time();
  for(unsigned int i = 0; i < current_order; ++i)
  {
    velocities.at(i + 1) = &get_velocity(i);
    times.at(i + 1)      = this->get_previous_time(i);
  }
}

template<int dim, typename Number>
void
TimeIntBDFCoupled<dim, Number>::do_timestep_solve()
{
  dealii::Timer timer;
  timer.restart();

  // extrapolate old solutions to obtain a good initial guess for the solver

  if(this->use_extrapolation)
  {
    solution_np.equ(this->extra.get_beta(0), solution[0]);
    for(unsigned int i = 1; i < solution.size(); ++i)
      solution_np.add(this->extra.get_beta(i), solution[i]);
  }

  // Linear problem
  {
    BlockVectorType rhs_vector;
    pde_operator->initialize_block_vector_velocity_pressure(rhs_vector);

    // calculate rhs vector
    pde_operator->rhs(rhs_vector, this->get_next_time());

    VectorType sum_alphai_ui(solution[0].block(0));

    // calculate Sum_i (alpha_i/dt * u_i)
    sum_alphai_ui.equ(this->bdf.get_alpha(0) / this->get_time_step_size(), solution[0].block(0));
    for(unsigned int i = 1; i < solution.size(); ++i)
    {
      sum_alphai_ui.add(this->bdf.get_alpha(i) / this->get_time_step_size(), solution[i].block(0));
    }

    // apply mass operator to sum_alphai_ui and add to rhs vector
    pde_operator->apply_mass_operator_add(rhs_vector.block(0), sum_alphai_ui);

    unsigned int const n_iter =
      pde_operator->solve(solution_np,
                          rhs_vector,
                          false,
                          this->get_next_time(),
                          this->get_scaling_factor_time_derivative_term());

    iterations.first += 1;
    iterations.second += n_iter;

    // write output
    if(this->print_solver_info())
    {
      this->pcout << std::endl << "Solve linear problem:";
      print_solver_info_linear(this->pcout, n_iter, timer.wall_time());
    }
  }

  this->timer_tree->insert({"Timeloop", "Coupled system"}, timer.wall_time());
}

template<int dim, typename Number>
void
TimeIntBDFCoupled<dim, Number>::solve_steady_problem()
{
}

template<int dim, typename Number>
void
TimeIntBDFCoupled<dim, Number>::prepare_vectors_for_next_timestep()
{
  push_back(solution);
  solution[0].swap(solution_np);
}

template<int dim, typename Number>
bool
TimeIntBDFCoupled<dim, Number>::print_solver_info() const
{
  return param.temporal_disc.solver_info_data.write(this->global_timer.wall_time(),
                                                              this->time - this->start_time,
                                                              this->time_step_number);
}

template<int dim, typename Number>
void
TimeIntBDFCoupled<dim, Number>::print_iterations() const
{
  print_list_of_iterations(this->pcout,
                           {"Coupled system"},
                           {static_cast<double>(iterations.second) /
                            std::max(1., static_cast<double>(iterations.first))});
}

template<int dim, typename Number>
void
TimeIntBDFCoupled<dim, Number>::postprocessing() const
{
  dealii::Timer timer;
  timer.restart();

  // pde_operator->distribute_constraint_u(const_cast<VectorType &>(get_velocity(0)));

  postprocessor->do_postprocessing(get_velocity(0),
                                   get_pressure(0),
                                   this->get_time(),
                                   this->get_time_step_number());

  this->timer_tree->insert({"Timeloop", "Postprocessing"}, timer.wall_time());
}

template class TimeIntBDFCoupled<2, float>;
template class TimeIntBDFCoupled<2, double>;
template class TimeIntBDFCoupled<3, float>;
template class TimeIntBDFCoupled<3, double>;

} // namespace Darcy
} // namespace ExaDG