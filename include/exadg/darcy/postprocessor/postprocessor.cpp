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

#include <exadg/darcy/postprocessor/postprocessor.h>

namespace ExaDG
{
namespace Darcy
{
template<int dim, typename Number>
PostProcessor<dim, Number>::PostProcessor(PostProcessorData<dim> const & postprocessor_data,
                                          MPI_Comm const &               comm)
  : mpi_comm(comm), pp_data(postprocessor_data), output_generator(comm)
{
}


template<int dim, typename Number>
void
PostProcessor<dim, Number>::setup(DarcyOperator const & pde_operator)
{
  darcy_operator = &pde_operator;

  output_generator.setup(pde_operator.get_dof_handler_u(),
                         pde_operator.get_dof_handler_p(),
                         *pde_operator.get_mapping(),
                         pp_data.output_data);
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::do_postprocessing(VectorType const &     velocity,
                                              VectorType const &     pressure,
                                              double const           time,
                                              types::time_step const time_step_number)
{
  /*
   *  write output
   */
  if(output_generator.time_control.needs_evaluation(time, time_step_number))
  {
    output_generator.evaluate(velocity, pressure, {}, time, false);
  }
}

template class PostProcessor<2, float>;
template class PostProcessor<2, double>;

template class PostProcessor<3, float>;
template class PostProcessor<3, double>;

} // namespace Darcy
} // namespace ExaDG
