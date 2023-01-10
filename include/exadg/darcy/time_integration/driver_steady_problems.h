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

#ifndef INCLUDE_EXADG_DARCY_TIME_INTEGRATION_DRIVER_STEADY_PROBLEMS_H_
#define INCLUDE_EXADG_DARCY_TIME_INTEGRATION_DRIVER_STEADY_PROBLEMS_H_

// deal.II
#include <deal.II/base/timer.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/utilities/timer_tree.h>

namespace ExaDG
{
// forward declarations
namespace IncNS
{
class Parameters;
} // namespace IncNS

namespace Darcy
{
template<int dim, typename Number>
class OperatorCoupled;

template<int dim, typename Number>
class DriverSteadyProblems
{
public:
  using VectorType      = dealii::LinearAlgebra::distributed::Vector<Number>;
  using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<Number>;

  using Operator = OperatorCoupled<dim, Number>;

  DriverSteadyProblems(std::shared_ptr<Operator>                              operator_,
                       IncNS::Parameters const &                              param_,
                       MPI_Comm const &                                       mpi_comm_,
                       std::shared_ptr<PostProcessorInterface<Number>> postprocessor_);

  void
  setup();

  void
  solve();

  VectorType const &
  get_velocity() const;

  std::shared_ptr<TimerTree>
  get_timings() const;

private:
  void
  initialize_vectors();

  void
  initialize_solution();

  void
  do_solve();

  void
  postprocessing() const;

  std::shared_ptr<Operator> pde_operator;

  IncNS::Parameters const & param;

  MPI_Comm const mpi_comm;

  bool is_test;

  dealii::Timer              global_timer;
  std::shared_ptr<TimerTree> timer_tree;

  dealii::ConditionalOStream pcout;

  BlockVectorType solution;
  BlockVectorType rhs_vector;

  std::shared_ptr<PostProcessorInterface<Number>> postprocessor;

  // iteration counts
  std::pair<
    unsigned int /* calls */,
    std::tuple<unsigned long long, unsigned long long> /* iteration counts {Newton, linear} */>
    iterations;
};
} // namespace Darcy
} // namespace ExaDG
#endif /* INCLUDE_EXADG_DARCY_TIME_INTEGRATION_DRIVER_STEADY_PROBLEMS_H_ */