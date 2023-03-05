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

#include <gtest/gtest.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/la_vector.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

namespace
{
unsigned int const dim = 2;
using Number           = double;

class CoupledInverseMassOperatorTest : public ::testing::Test
{
private:
  static unsigned int const degree               = 1;
  static unsigned int const n_global_refinements = 0;

protected:
  using Integrator = dealii::FEEvaluation<dim, -1, 0, dim, Number, dealii::VectorizedArray<Number>>;
  using CellWiseInverseMass =
    dealii::MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, dim, Number>;

  void
  SetUp() override
  {
    dealii::GridGenerator::hyper_cube(triangulation, 0.0, 1.0);
    triangulation.refine_global(n_global_refinements);

    fe = std::make_unique<dealii::FESystem<dim>>(dealii::FE_DGQ<dim>(degree), dim);

    dof_handler.reinit(triangulation);
    dof_handler.distribute_dofs(*fe);

    constraints.close();

    mapping    = std::make_unique<dealii::MappingQ<dim>>(degree);
    quadrature = std::make_unique<dealii::QGauss<dim>>(degree + 1);

    matrix_free.reinit(*mapping, dof_handler, constraints, *quadrature);
  }

  std::unique_ptr<dealii::FESystem<dim>> fe;
  std::unique_ptr<dealii::MappingQ<dim>> mapping;
  std::unique_ptr<dealii::QGauss<dim>>   quadrature;

  dealii::AffineConstraints<Number> constraints;
  dealii::Triangulation<dim>        triangulation;
  dealii::DoFHandler<dim>           dof_handler;
  dealii::MatrixFree<dim, Number>   matrix_free;
};

TEST_F(CoupledInverseMassOperatorTest, Solves)
{
  Integrator          integrator(matrix_free, 0, 0);
  CellWiseInverseMass inverse(integrator);

  unsigned int const n_dofs               = matrix_free.get_dof_handler(0).n_dofs();
  unsigned int const n_dofs_per_component = n_dofs / dim;

  dealii::LinearAlgebra::Vector<Number> src(n_dofs);
  dealii::LinearAlgebra::Vector<Number> dst(n_dofs);

  integrator.reinit(0);

  src.add(1.0);
  integrator.read_dof_values(src, 0);

  dealii::AlignedVector<dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>> coefficients(
    n_dofs_per_component);

  dealii::AlignedVector<dealii::VectorizedArray<Number>> inv_JxW_values(n_dofs_per_component);
  inverse.fill_inverse_JxW_values(inv_JxW_values);

  for(unsigned int q = 0; q < n_dofs_per_component; ++q)
    coefficients[q] = 3.0 * inv_JxW_values[q] *
                      dealii::unit_symmetric_tensor<dim, dealii::VectorizedArray<Number>>();

  inverse.apply(coefficients, integrator.begin_dof_values(), integrator.begin_dof_values());
  integrator.set_dof_values(dst, 0);

  for(double const val : dst)
    EXPECT_NEAR(val, 12.0, 1.0e-12);
}

} // namespace