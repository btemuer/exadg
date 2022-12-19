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

#ifndef INCLUDE_EXADG_DARCY_POSTPROCESSOR_POSTPROCESSOR_H_
#define INCLUDE_EXADG_DARCY_POSTPROCESSOR_POSTPROCESSOR_H_

#include <exadg/darcy/postprocessor/postprocessor_interface.h>
#include <exadg/darcy/spatial_discretization/operator_coupled.h>
#include <exadg/incompressible_navier_stokes/postprocessor/output_generator.h>

namespace ExaDG
{
namespace Darcy
{
template<int dim>
struct PostProcessorData
{
  IncNS::OutputData output_data;
};

template<int dim, typename Number>
class PostProcessor : public PostProcessorInterface<Number>
{
public:
  using DarcyOperator = OperatorCoupled<dim, Number>;

  using VectorType = typename PostProcessorInterface<Number>::VectorType;

  PostProcessor(PostProcessorData<dim> const & postprocessor_data, MPI_Comm const & mpi_comm);

  virtual ~PostProcessor() = default;

  void
  setup(DarcyOperator const & pde_operator);

  void
  do_postprocessing(VectorType const &     velocity,
                    VectorType const &     pressure,
                    double const           time             = 0.0,
                    types::time_step const time_step_number = numbers::steady_timestep) override;

protected:
  MPI_Comm const mpi_comm;

private:
  PostProcessorData<dim> pp_data;

  dealii::SmartPointer<DarcyOperator const> darcy_operator;

  // write output for visualization of results (e.g., using paraview)
  IncNS::OutputGenerator<dim, Number> output_generator;
};



} // namespace Darcy
} // namespace ExaDG

#endif /* INCLUDE_EXADG_DARCY_POSTPROCESSOR_POSTPROCESSOR_H_ */
