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

#ifndef INCLUDE_CONVECTION_DIFFUSION_MULTIGRID_PRECONDITIONER_H_
#define INCLUDE_CONVECTION_DIFFUSION_MULTIGRID_PRECONDITIONER_H_

#include <exadg/convection_diffusion/spatial_discretization/operators/combined_operator.h>
#include <exadg/operators/multigrid_operator.h>
#include <exadg/solvers_and_preconditioners/multigrid/multigrid_preconditioner_base.h>

namespace ExaDG
{
namespace ConvDiff
{
/*
 *  Multigrid preconditioner for scalar convection-diffusion equation.
 */
template<int dim, typename Number>
class MultigridPreconditioner : public MultigridPreconditionerBase<dim, Number>
{
private:
  typedef MultigridPreconditionerBase<dim, Number> Base;

public:
  typedef typename Base::MultigridNumber MultigridNumber;

private:
  typedef CombinedOperator<dim, Number>          PDEOperator;
  typedef CombinedOperator<dim, MultigridNumber> PDEOperatorMG;

  typedef MultigridOperatorBase<dim, MultigridNumber>            MGOperatorBase;
  typedef MultigridOperator<dim, MultigridNumber, PDEOperatorMG> MGOperator;

  typedef typename Base::Map_DBC               Map_DBC;
  typedef typename Base::Map_DBC_ComponentMask Map_DBC_ComponentMask;
  typedef typename Base::PeriodicFacePairs     PeriodicFacePairs;
  typedef typename Base::VectorType            VectorType;
  typedef typename Base::VectorTypeMG          VectorTypeMG;

public:
  MultigridPreconditioner(MPI_Comm const & mpi_comm);

  virtual ~MultigridPreconditioner(){};

  /*
   *  This function initializes the multigrid preconditioner.
   */
  void
  initialize(
    MultigridData const &                                                  mg_data,
    MultigridVariant const &                                               multigrid_variant,
    dealii::Triangulation<dim> const *                                     triangulation,
    PeriodicFacePairs const &                                              periodic_face_pairs,
    std::vector<std::shared_ptr<dealii::Triangulation<dim> const>> const & coarse_triangulations,
    std::vector<PeriodicFacePairs> const &      coarse_periodic_face_pairs,
    dealii::FiniteElement<dim> const &          fe,
    std::shared_ptr<dealii::Mapping<dim> const> mapping,
    PDEOperator const &                         pde_operator,
    MultigridOperatorType const &               mg_operator_type,
    bool const                                  mesh_is_moving,
    Map_DBC const &                             dirichlet_bc,
    Map_DBC_ComponentMask const &               dirichlet_bc_component_mask);

  /*
   *  This function updates the multigrid preconditioner.
   */
  void
  update() override;

private:
  void
  fill_matrix_free_data(MatrixFreeData<dim, MultigridNumber> & matrix_free_data,
                        unsigned int const                     level,
                        unsigned int const                     h_level) override;

  std::shared_ptr<MGOperatorBase>
  initialize_operator(unsigned int const level) override;

  void
  initialize_dof_handler_and_constraints(
    bool const                         operator_is_singular,
    dealii::FiniteElement<dim> const & fe,
    Map_DBC const &                    dirichlet_bc,
    Map_DBC_ComponentMask const &      dirichlet_bc_component_mask) override;

  void
  initialize_transfer_operators() override;

  /*
   *  This function updates the operators on all levels
   */
  void
  update_operators();

  /*
   * This function updates the velocity field for all levels.
   * In order to update mg_matrices[level] this function has to be called.
   */
  void
  set_velocity(VectorTypeMG const & velocity);

  /*
   * This function performs the updates that are necessary after the mesh has been moved
   * and after matrix_free has been updated.
   */
  void
  update_operators_after_mesh_movement();

  /*
   *  This function sets the current the time.
   *  In order to update operators[level] this function has to be called.
   *  (This is due to the fact that the velocity field of the convective term
   *  is a function of the time.)
   */
  void
  set_time(double const & time);

  /*
   *  This function updates the scaling factor of the mass operator.
   *  In order to update operators[level] this function has to be called.
   *  This is necessary if adaptive time stepping is used where
   *  the scaling factor of the derivative term is variable.
   */
  void
  set_scaling_factor_mass_operator(double const & scaling_factor);

  /*
   *  This function updates the smoother for all levels of the multigrid
   *  algorithm.
   *  The prerequisite to call this function is that operators[level] have
   *  been updated.
   */
  void
  update_smoothers();

  std::shared_ptr<PDEOperatorMG>
  get_operator(unsigned int level) const;

  std::shared_ptr<MGTransfer<VectorTypeMG>> transfers_velocity;

  dealii::MGLevelObject<std::shared_ptr<dealii::DoFHandler<dim> const>> dof_handlers_velocity;
  dealii::MGLevelObject<std::shared_ptr<dealii::MGConstrainedDoFs>>     constrained_dofs_velocity;
  dealii::MGLevelObject<std::shared_ptr<dealii::AffineConstraints<MultigridNumber>>>
    constraints_velocity;

  CombinedOperatorData<dim> data;

  PDEOperator const * pde_operator;

  MultigridOperatorType mg_operator_type;

  bool mesh_is_moving;
};

} // namespace ConvDiff
} // namespace ExaDG

#endif /* INCLUDE_CONVECTION_DIFFUSION_MULTIGRID_PRECONDITIONER_H_ */
