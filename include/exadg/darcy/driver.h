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

#ifndef INCLUDE_EXADG_DARCY_DRIVER_H_
#define INCLUDE_EXADG_DARCY_DRIVER_H_

#include <exadg/darcy/postprocessor/postprocessor.h>
#include <exadg/darcy/spatial_discretization/operator_coupled.h>
#include <exadg/darcy/time_integration/driver_steady_problems.h>
#include <exadg/darcy/time_integration/time_int_bdf_coupled.h>
#include <exadg/darcy/user_interface/application_base.h>
#include <exadg/grid/grid_motion_function.h>
#include <exadg/matrix_free/matrix_free_data.h>
#include <exadg/utilities/print_general_infos.h>

namespace ExaDG
{
namespace Darcy
{
inline unsigned int
get_dofs_per_element(std::string const & input_file,
                     unsigned int const  dim,
                     unsigned int const  degree)
{
  std::string operator_type_string, pressure_degree = "MixedOrder";

  dealii::ParameterHandler prm;
  // clang-format off
  prm.enter_subsection("Discretization");
    prm.add_parameter("PressureDegree",
                      pressure_degree,
                      "Degree of pressure shape functions.",
                      dealii::Patterns::Selection("MixedOrder|EqualOrder"),
                      true);
  prm.leave_subsection();
  // clang-format on
  prm.parse_input(input_file, "", true, true);

  unsigned int const velocity_dofs_per_element = dim * dealii::Utilities::pow(degree + 1, dim);

  unsigned int pressure_dofs_per_element = 1;

  if(pressure_degree == "MixedOrder")
    pressure_dofs_per_element = dealii::Utilities::pow(degree, dim);
  else if(pressure_degree == "EqualOrder")
    pressure_dofs_per_element = dealii::Utilities::pow(degree + 1, dim);
  else
    AssertThrow(false, dealii::ExcNotImplemented());

  return velocity_dofs_per_element + pressure_dofs_per_element;
}

template<int dim, typename Number>
class Driver
{
public:
  Driver(MPI_Comm const & mpi_comm, std::shared_ptr<ApplicationBase<dim, Number>> application);

  void
  setup();

  void
  ale_update() const;

  void
  solve() const;

  void
  print_performance_results(double total_time) const;

private:
  // MPI communicator
  MPI_Comm const mpi_comm;

  // output to std::cout
  dealii::ConditionalOStream pcout;

  // application
  std::shared_ptr<ApplicationBase<dim, Number>> application;

  /*
   * MatrixFree
   */
  std::shared_ptr<MatrixFreeData<dim, Number>>     matrix_free_data;
  std::shared_ptr<dealii::MatrixFree<dim, Number>> matrix_free;

  /*
   * Spatial discretization
   */
  std::shared_ptr<OperatorCoupled<dim, Number>> pde_operator;

  // moving mapping (ALE)
  std::shared_ptr<GridMotionBase<dim, Number>> grid_motion;

  /*
   * Postprocessor
   */
  using Postprocessor = PostProcessor<dim, Number>;

  std::shared_ptr<Postprocessor> postprocessor;

  /*
   * Temporal discretization
   */

  // unsteady solver
  std::shared_ptr<TimeIntBDFCoupled<dim, Number>> time_integrator;

  // steady solver
  std::shared_ptr<DriverSteadyProblems<dim, Number>> driver_steady;

  /*
   * Computation time (wall clock time)
   */
  mutable TimerTree timer_tree;
};

} // namespace Darcy
} // namespace ExaDG

#endif /* INCLUDE_EXADG_DARCY_DRIVER_H_ */