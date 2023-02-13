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

#ifndef INCLUDE_EXADG_DARCY_TIME_INTEGRATION_TIME_INT_BDF_H_
#define INCLUDE_EXADG_DARCY_TIME_INTEGRATION_TIME_INT_BDF_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/time_integration/time_int_bdf_base.h>

namespace ExaDG
{
// forward declaration
namespace IncNS
{
class Parameters;
} // namespace IncNS

namespace Darcy
{
// forward declaration
template<int dim, typename Number>
class OperatorCoupled;

template<typename Number>
class PostProcessorInterface;

template<int dim, typename Number>
class TimeIntBDFCoupled : public TimeIntBDFBase<Number>
{
  using Base = TimeIntBDFBase<Number>;

  using VectorType      = typename Base::VectorType;
  using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<Number>;

  using Operator = OperatorCoupled<dim, Number>;

public:
  TimeIntBDFCoupled(std::shared_ptr<Operator>                       operator_in,
                    Parameters const &                       param_in,
                    MPI_Comm const &                                mpi_comm_in,
                    std::shared_ptr<PostProcessorInterface<Number>> postprocessor_in);

  void
  print_iterations() const;

  bool
  print_solver_info() const final;

  double
  calculate_time_step_size() final;

  double
  recalculate_time_step_size() const final;

  void
  ale_update();

  VectorType const &
  get_velocity() const;

  VectorType const &
  get_velocity_np() const;

  VectorType const &
  get_pressure() const;

  VectorType const &
  get_pressure_np() const;

  void
  get_velocities_and_times(std::vector<VectorType const *> & velocities,
                           std::vector<double> &             times) const;

  void
  get_velocities_and_times_np(std::vector<VectorType const *> & velocities,
                              std::vector<double> &             times) const;

private:
  void
  allocate_vectors() final;

  void
  setup_derived() final;

  void
  read_restart_vectors(boost::archive::binary_iarchive & ia) final;

  void
  write_restart_vectors(boost::archive::binary_oarchive & oa) const final;

  void
  initialize_current_solution() final;

  void
  initialize_former_solutions() final;

  void
  do_timestep_solve() final;

  void
  solve_steady_problem();

  void
  prepare_vectors_for_next_timestep() final;

  VectorType const &
  get_velocity(unsigned int i /* t_{n-i} */) const;

  VectorType const &
  get_pressure(unsigned int i /* t_{n-i} */) const;

  void
  set_velocity(VectorType const & velocity, unsigned int const i /* t_{n-i} */);

  void
  set_pressure(VectorType const & pressure, unsigned int const i /* t_{n-i} */);

  void
  postprocessing() const final;

  Parameters const & param;

  bool use_extrapolation;

  std::shared_ptr<Operator> pde_operator;

  std::vector<BlockVectorType> solution;
  BlockVectorType              solution_np;

  // iteration counts
  std::pair<
    unsigned int /* calls */,
    unsigned long long> /* iteration counts (linear) */
    iterations;

  // postprocessor
  std::shared_ptr<PostProcessorInterface<Number>> postprocessor;

  // ALE
  VectorType              grid_velocity;
  std::vector<VectorType> vec_grid_coordinates;
  VectorType              grid_coordinates_np;
  VectorType              grid_displacements_np;
};

} // namespace Darcy
} // namespace ExaDG


#endif // INCLUDE_EXADG_DARCY_TIME_INTEGRATION_TIME_INT_BDF_H_
