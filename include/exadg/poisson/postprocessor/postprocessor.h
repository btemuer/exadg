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

#ifndef INCLUDE_POISSON_POSTPROCESSOR_H_
#define INCLUDE_POISSON_POSTPROCESSOR_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/poisson/postprocessor/postprocessor_base.h>
#include <exadg/postprocessor/error_calculation.h>
#include <exadg/postprocessor/normal_flux_calculation.h>
#include <exadg/postprocessor/output_data_base.h>
#include <exadg/postprocessor/output_generator_scalar.h>

namespace ExaDG
{
namespace Poisson
{
template<int dim>
struct PostProcessorData
{
  PostProcessorData()
  {
  }

  OutputDataBase            output_data;
  ErrorCalculationData<dim> error_data;
  NormalFluxCalculatorData  normal_flux_data;
};

template<int dim, int n_components, typename Number>
class PostProcessor : public PostProcessorBase<dim, n_components, Number>
{
protected:
  typedef PostProcessorBase<dim, n_components, Number> Base;

  typedef typename Base::VectorType VectorType;

public:
  PostProcessor(PostProcessorData<dim> const & pp_data, MPI_Comm const & mpi_comm);

  void
  setup(Operator<dim, n_components, Number> const & pde_operator) override;

  void
  do_postprocessing(VectorType const &     solution,
                    double const           time             = 0.0,
                    types::time_step const time_step_number = numbers::steady_timestep) override;

protected:
  MPI_Comm const mpi_comm;

private:
  PostProcessorData<dim> pp_data;

  OutputGenerator<dim, Number> output_generator;
  ErrorCalculator<dim, Number> error_calculator;

  std::shared_ptr<NormalFluxCalculator<dim, Number>> normal_flux_calculator;
};

} // namespace Poisson
} // namespace ExaDG


#endif /* INCLUDE_POISSON_POSTPROCESSOR_H_ */
