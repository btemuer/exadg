#include "multigrid_preconditioner_base.h"

#include <navierstokes/config.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <map>
#include <vector>

#include "../transfer/mg_transfer_mf_c.h"
#include "../transfer/mg_transfer_mf_h.h"
#include "../transfer/mg_transfer_mf_p.h"

#include "../mg_coarse/mg_coarse_ml.h"

#include "../util/compute_eigenvalues.h"

template<int dim, typename Number, typename MultigridNumber>
MultigridPreconditionerBase<dim, Number, MultigridNumber>::MultigridPreconditionerBase(
  std::shared_ptr<Operator> multigrid_operator)
  : underlying_operator(multigrid_operator)
{
}

template<int dim, typename Number, typename MultigridNumber>
MultigridPreconditionerBase<dim, Number, MultigridNumber>::~MultigridPreconditionerBase()
{
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::initialize(
  MultigridData const &   mg_data,
  DoFHandler<dim> const & dof_handler,
  Mapping<dim> const &    mapping,
  void *                  operator_data,
  Map const *             dirichlet_bc_in,
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> *
                          periodic_face_pairs_in,
  DoFHandler<dim> const * add_dof_handler)
{
  this->mg_data = mg_data;

  // get triangulation
  parallel::Triangulation<dim> const * tria =
    dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler.get_triangulation());

  if((/*is_cg ||*/ mg_data.coarse_solver == MultigridCoarseGridSolver::AMG_ML) &&
     ((dirichlet_bc_in == nullptr) || (dirichlet_bc_in == nullptr)))
    AssertThrow(
      mg_data.coarse_solver != MultigridCoarseGridSolver::AMG_ML,
      ExcMessage(
        "You have to provide Dirichlet BCs and peridic face pairs if you want to use CG or AMG!"));

  // dereference points
  auto & dirichlet_bc        = *dirichlet_bc_in;
  auto & periodic_face_pairs = *periodic_face_pairs_in;

  // extract paramters
  const auto   mg_type = this->mg_data.type;
  unsigned int degree  = dof_handler.get_fe().degree;
  const bool   is_dg   = dof_handler.get_fe().dofs_per_vertex == 0;

  // setup sequence
  std::vector<unsigned int>           h_levels;
  std::vector<MGDofHandlerIdentifier> p_levels;
  std::vector<MGLevelIdentifier>      global_levels;
  this->initialize_mg_sequence(tria, global_levels, h_levels, p_levels, degree, mg_type, is_dg);
  this->check_mg_sequence(global_levels);
  this->n_global_levels = global_levels.size(); // number of actual multigrid levels

  // setup of multigrid components
  this->initialize_mg_dof_handler_and_constraints(
    dof_handler, tria, global_levels, p_levels, dirichlet_bc);
  this->initialize_mg_matrices(
    global_levels, mapping, periodic_face_pairs, operator_data, add_dof_handler);
  this->initialize_smoothers();
  this->initialize_coarse_solver(global_levels[0].level);
  this->initialize_mg_transfer(dof_handler.get_fe().n_components(),
                               Utilities::MPI::this_mpi_process(tria->get_communicator()),
                               global_levels,
                               p_levels,
                               mg_matrices,
                               mg_dofhandler,
                               mg_constrained_dofs,
                               mg_transfer.mg_level_object);

  this->initialize_multigrid_preconditioner();
}

/*
 *
 * example: h_levels = [0 1 2], p_levels = [1 3 7]
 *
 * p-MG:
 * global_levels  h_levels  p_levels
 * 2              2         7
 * 1              2         3
 * 0              2         1
 *
 * ph-MG:
 * global_levels  h_levels  p_levels
 * 4              2         7
 * 3              2         3
 * 2              2         1
 * 1              1         1
 * 0              0         1
 *
 * h-MG:
 * global_levels  h_levels  p_levels
 * 2              2         7
 * 1              1         7
 * 0              0         7
 *
 * hp-MG:
 * global_levels  h_levels  p_levels
 * 4              2         7
 * 3              1         7
 * 2              0         7
 * 1              0         3
 * 0              0         1
 *
 */

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::initialize_mg_sequence(
  const parallel::Triangulation<dim> *  tria,
  std::vector<MGLevelIdentifier> &      global_levels,
  std::vector<unsigned int> &           h_levels,
  std::vector<MGDofHandlerIdentifier> & p_levels,
  unsigned int                          degree,
  MultigridType                         mg_type,
  const bool                            is_dg)
{
  // setup h-levels
  if(mg_type == MultigridType::pMG) // p-MG is only working on the finest h-level
  {
    h_levels.push_back(tria->n_global_levels() - 1);
  }
  else // h-MG, hp-MG, and ph-MG are working on all h-levels
  {
    for(unsigned int i = 0; i < tria->n_global_levels(); i++)
      h_levels.push_back(i);
  }

  // setup p-levels
  if(mg_type == MultigridType::hMG) // h-MG is only working on high-order
  {
    p_levels.push_back({degree, is_dg});
  }
  else // p-MG, hp-MG, and ph-MG are working on high- and low- order elements
  {
    unsigned int temp = degree;
    do
    {
      p_levels.push_back({temp, is_dg});
      switch(this->mg_data.p_sequence)
      {
          // clang-format off
        case PSequenceType::GO_TO_ONE:       temp = 1;                                                break;
        case PSequenceType::DECREASE_BY_ONE: temp = std::max(temp-1, 1u);                             break;
        case PSequenceType::BISECTION:       temp = std::max(temp/2, 1u);                             break;
        case PSequenceType::MANUAL:          temp = (degree==3&&temp==3) ? 2 : std::max(degree/2, 1u);break;
        default:
          AssertThrow(false, ExcMessage("No valid p-sequence selected!"));
          // clang-format on
      }
    } while(temp != p_levels.back().degree);
    std::reverse(std::begin(p_levels), std::end(p_levels));
  }

  AssertThrow(!(mg_data.c_transfer_front && mg_data.c_transfer_back),
              ExcMessage("You can only use c_transfer once!"));

  if(mg_data.c_transfer_back && is_dg)
    p_levels.insert(p_levels.begin(), {p_levels.front().degree, false});

  if(mg_data.c_transfer_front && is_dg)
  {
    for(auto & i : p_levels)
      i.is_dg = false;
    p_levels.push_back({p_levels.back().degree, true});
  }

  // setup global-levels
  if(mg_type == MultigridType::pMG || mg_type == MultigridType::phMG)
  {
    // top level: p-gmg
    if(mg_type == MultigridType::phMG) // low level: h-gmg
      for(unsigned int i = 0; i < h_levels.size() - 1; i++)
        global_levels.push_back({h_levels[i], p_levels.front()});
    for(auto deg : p_levels)
      global_levels.push_back({h_levels.back(), deg});
  }
  else if(mg_type == MultigridType::hMG || mg_type == MultigridType::hpMG)
  {
    // top level: h-gmg
    if(mg_type == MultigridType::hpMG) // low level: p-gmg
      for(unsigned int i = 0; i < p_levels.size() - 1; i++)
        global_levels.push_back({h_levels.front(), p_levels[i]});
    for(auto geo : h_levels)
      global_levels.push_back({geo, p_levels.back()});
  }
  else
    AssertThrow(false, ExcMessage("This multigrid type does not exist!"));

  this->n_global_levels = global_levels.size(); // number of actual multigrid levels

  this->check_mg_sequence(global_levels);
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::check_mg_sequence(
  std::vector<MGLevelIdentifier> const & global_levels)
{
  AssertThrow(this->n_global_levels == global_levels.size(),
              ExcMessage("Variable n_global_levels is not initialized correctly."));

  for(unsigned int i = 1; i < global_levels.size(); i++)
  {
    auto fine_level   = global_levels[i];
    auto coarse_level = global_levels[i - 1];

    AssertThrow((fine_level.level != coarse_level.level) ^
                  (fine_level.degree != coarse_level.degree) ^
                  (fine_level.is_dg != coarse_level.is_dg),
                ExcMessage(
                  "Between levels there is only ONE change allowed: either in h- or p-level!"));
  }
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::
  initialize_mg_dof_handler_and_constraints(
    DoFHandler<dim> const &                                              dof_handler,
    parallel::Triangulation<dim> const *                                 tria,
    std::vector<MGLevelIdentifier> &                                     global_levels,
    std::vector<MGDofHandlerIdentifier> &                                p_levels,
    std::map<types::boundary_id, std::shared_ptr<Function<dim>>> const & dirichlet_bc)
{
  this->mg_constrained_dofs.resize(0, this->n_global_levels - 1);
  this->mg_dofhandler.resize(0, this->n_global_levels - 1);

  const unsigned int n_components = dof_handler.get_fe().n_components();

  // temporal storage for new dofhandlers and constraints on each p-level
  std::map<MGDofHandlerIdentifier, std::shared_ptr<const DoFHandler<dim>>> map_dofhandlers;
  std::map<MGDofHandlerIdentifier, std::shared_ptr<MGConstrainedDoFs>>     map_constraints;

  // setup dof-handler and constrained dofs for each p-level
  for(auto degree : p_levels)
  {
    // setup dof_handler: create dof_handler...
    auto dof_handler = new DoFHandler<dim>(*tria);
    // ... create FE and distribute it
    if(degree.is_dg)
      dof_handler->distribute_dofs(FESystem<dim>(FE_DGQ<dim>(degree.degree), n_components));
    else
      dof_handler->distribute_dofs(FESystem<dim>(FE_Q<dim>(degree.degree), n_components));
    dof_handler->distribute_mg_dofs();
    // setup constrained dofs:
    auto constrained_dofs = new MGConstrainedDoFs();
    constrained_dofs->clear();
    this->initialize_mg_constrained_dofs(*dof_handler, *constrained_dofs, dirichlet_bc);

    // put in temporal storage
    map_dofhandlers[degree] = std::shared_ptr<DoFHandler<dim> const>(dof_handler);
    map_constraints[degree] = std::shared_ptr<MGConstrainedDoFs>(constrained_dofs);
  }

  // populate dof-handler and constrained dofs to all hp-levels with the same degree
  for(unsigned int i = 0; i < global_levels.size(); i++)
  {
    auto degree            = global_levels[i].id;
    mg_dofhandler[i]       = map_dofhandlers[degree];
    mg_constrained_dofs[i] = map_constraints[degree];
  }
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::initialize_mg_matrices(
  std::vector<MGLevelIdentifier> & global_levels,
  const Mapping<dim> &             mapping,
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
                          periodic_face_pairs,
  void *                  operator_data_in,
  DoFHandler<dim> const * add_dof_handler)
{
  this->mg_matrices.resize(0, this->n_global_levels - 1);

  // create and setup operator on each level
  for(unsigned int i = 0; i < this->n_global_levels; i++)
  {
    auto matrix = static_cast<Operator *>(underlying_operator->get_new(global_levels[i].degree));

    if(add_dof_handler != nullptr)
    {
      matrix->reinit_multigrid_add_dof_handler(*mg_dofhandler[i],
                                               mapping,
                                               operator_data_in,
                                               *this->mg_constrained_dofs[i],
                                               periodic_face_pairs,
                                               global_levels[i].level,
                                               add_dof_handler);
    }
    else
    {
      matrix->reinit_multigrid(*mg_dofhandler[i],
                               mapping,
                               operator_data_in,
                               *this->mg_constrained_dofs[i],
                               periodic_face_pairs,
                               global_levels[i].level);
    }
    mg_matrices[i].reset(matrix);
  }
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::initialize_smoothers()
{
  this->mg_smoother.resize(0, this->n_global_levels - 1);

  for(unsigned int i = 1; i < this->n_global_levels; i++)
    this->initialize_smoother(*this->mg_matrices[i], i);
}


template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::initialize_mg_transfer(
  const int                                                      n_components,
  const int                                                      rank,
  std::vector<MGLevelIdentifier> &                               global_levels,
  std::vector<MGDofHandlerIdentifier> &                          p_levels,
  MGLevelObject<std::shared_ptr<Operator>> &                     mg_matrices,
  MGLevelObject<std::shared_ptr<const DoFHandler<dim>>> &        mg_dofhandler,
  MGLevelObject<std::shared_ptr<MGConstrainedDoFs>> &            mg_constrained_dofs,
  MGLevelObject<std::shared_ptr<MGTransferBase<VectorTypeMG>>> & mg_transfer)
{
  mg_transfer.resize(0, global_levels.size() - 1);

#ifndef DEBUG
  (void)rank; // avoid compiler warning
#endif

  std::map<MGDofHandlerIdentifier, std::shared_ptr<MGTransferMFH<dim, MultigridNumber>>>
    mg_tranfers_temp;
  std::map<MGDofHandlerIdentifier, std::map<unsigned int, unsigned int>>
    map_global_level_to_h_levels;

  // initialize maps so that we do not have to check existence later on
  for(auto deg : p_levels)
    map_global_level_to_h_levels[deg] = {};

  // fill the maps
  for(unsigned int i = 0; i < global_levels.size(); i++)
  {
    auto level = global_levels[i];

    map_global_level_to_h_levels[level.id][i] = level.level;
  }

  // create h-transfer operators between levels
  for(auto deg : p_levels)
  {
    if(map_global_level_to_h_levels[deg].size() > 1)
    {
      // create actual h-transfer-operator
      std::shared_ptr<MGTransferMFH<dim, MultigridNumber>> transfer(
        new MGTransferMFH<dim, MultigridNumber>(map_global_level_to_h_levels[deg]));

      // dof-handlers and constrains are saved for global levels
      // so we have to convert degree to any global level which has this degree
      // (these share the same dof-handlers and constraints)
      unsigned int global_level = map_global_level_to_h_levels[deg].begin()->first;
      transfer->initialize_constraints(*mg_constrained_dofs[global_level]);
      transfer->build(*mg_dofhandler[global_level]);
      mg_tranfers_temp[deg] = transfer;
    } // else: there is only one global level (and one h-level) on this p-level
  }

  // fill mg_transfer with the correct transfers
  for(unsigned int i = 1; i < global_levels.size(); i++)
  {
    auto coarse_level = global_levels[i - 1];
    auto fine_level   = global_levels[i];

    std::shared_ptr<MGTransferBase<VectorTypeMG>> temp;

    if(coarse_level.level != fine_level.level) // h-transfer
    {
#ifdef DEBUG
      if(rank == 0)
        printf("  h-MG (l=%2d,k=%2d) -> (l=%2d,k=%2d)\n",
               coarse_level.level,
               coarse_level.degree,
               fine_level.level,
               fine_level.degree);
#endif

      temp = mg_tranfers_temp[coarse_level.id]; // get the previously h-transfer operator
    }
    else if(coarse_level.degree != fine_level.degree) // p-transfer
    {
#ifdef DEBUG
      if(rank == 0)
        printf("  p-MG (l=%2d,k=%2d) -> (l=%2d,k=%2d)\n",
               coarse_level.level,
               coarse_level.degree,
               fine_level.level,
               fine_level.degree);
#endif

      if(n_components == 1)
        temp.reset(
          new MGTransferMFP<dim, MultigridNumber, VectorTypeMG, 1>(&mg_matrices[i]->get_data(),
                                                                   &mg_matrices[i - 1]->get_data(),
                                                                   fine_level.degree,
                                                                   coarse_level.degree));
      else if(n_components == dim)
        temp.reset(new MGTransferMFP<dim, MultigridNumber, VectorTypeMG, dim>(
          &mg_matrices[i]->get_data(),
          &mg_matrices[i - 1]->get_data(),
          fine_level.degree,
          coarse_level.degree));
      else
        AssertThrow(false, ExcMessage("Cannot create MGTransferMFP!"));
    }
    else if(coarse_level.is_dg != fine_level.is_dg) // c-transfer
    {
#ifdef DEBUG
      if(rank == 0)
        printf("  c-MG (l=%2d,k=%2d) -> (l=%2d,k=%2d)\n",
               coarse_level.level,
               coarse_level.degree,
               fine_level.level,
               fine_level.degree);
#endif

      if(n_components == 1)
        temp.reset(new MGTransferMFC<dim, typename Operator::value_type, VectorTypeMG, 1>(
          mg_matrices[i]->get_data(),
          mg_matrices[i - 1]->get_data(),
          mg_matrices[i]->get_constraint_matrix(),
          mg_matrices[i - 1]->get_constraint_matrix(),
          fine_level.level,
          coarse_level.degree));
      else if(n_components == dim)
        temp.reset(new MGTransferMFC<dim, typename Operator::value_type, VectorTypeMG, dim>(
          mg_matrices[i]->get_data(),
          mg_matrices[i - 1]->get_data(),
          mg_matrices[i]->get_constraint_matrix(),
          mg_matrices[i - 1]->get_constraint_matrix(),
          fine_level.level,
          coarse_level.degree));
      else
        AssertThrow(false, ExcMessage("Cannot create MGTransferMFP!"));
    }
    mg_transfer[i] = temp;
  }
}


template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::initialize_mg_constrained_dofs(
  DoFHandler<dim> const & dof_handler,
  MGConstrainedDoFs &     constrained_dofs,
  Map const &             dirichlet_bc)
{
  std::set<types::boundary_id> dirichlet_boundary;
  for(auto & it : dirichlet_bc)
    dirichlet_boundary.insert(it.first);
  constrained_dofs.initialize(dof_handler);
  constrained_dofs.make_zero_boundary_constraints(dof_handler, dirichlet_boundary);
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::update(
  LinearOperatorBase const * /*linear_operator*/)
{
  // do nothing in base class (has to be implemented by derived classes if necessary)
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::vmult(VectorType &       dst,
                                                                 VectorType const & src) const
{
  multigrid_preconditioner->vmult(dst, src);
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::apply_smoother_on_fine_level(
  VectorTypeMG &       dst,
  VectorTypeMG const & src) const
{
  this->mg_smoother[this->mg_smoother.max_level()]->vmult(dst, src);
}


template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::update_smoother(unsigned int level)
{
  AssertThrow(level > 0,
              ExcMessage("Multigrid level is invalid when initializing multigrid smoother!"));

  switch(mg_data.smoother)
  {
    case MultigridSmoother::Chebyshev:
    {
      initialize_chebyshev_smoother(*mg_matrices[level], level);
      break;
    }
    case MultigridSmoother::ChebyshevNonsymmetricOperator:
    {
      initialize_chebyshev_smoother_nonsymmetric_operator(*mg_matrices[level], level);
      break;
    }
    case MultigridSmoother::GMRES:
    {
      typedef GMRESSmoother<Operator, VectorTypeMG> GMRES_SMOOTHER;

      std::shared_ptr<GMRES_SMOOTHER> smoother =
        std::dynamic_pointer_cast<GMRES_SMOOTHER>(mg_smoother[level]);
      smoother->update();
      break;
    }
    case MultigridSmoother::CG:
    {
      typedef CGSmoother<Operator, VectorTypeMG> CG_SMOOTHER;

      std::shared_ptr<CG_SMOOTHER> smoother =
        std::dynamic_pointer_cast<CG_SMOOTHER>(mg_smoother[level]);
      smoother->update();
      break;
    }
    case MultigridSmoother::Jacobi:
    {
      typedef JacobiSmoother<Operator, VectorTypeMG> JACOBI_SMOOTHER;

      std::shared_ptr<JACOBI_SMOOTHER> smoother =
        std::dynamic_pointer_cast<JACOBI_SMOOTHER>(mg_smoother[level]);
      smoother->update();
      break;
    }
    default:
    {
      AssertThrow(false, ExcMessage("Specified MultigridSmoother not implemented!"));
    }
  }
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::update_coarse_solver()
{
  switch(mg_data.coarse_solver)
  {
    case MultigridCoarseGridSolver::Chebyshev:
    {
      initialize_chebyshev_smoother_coarse_grid(*mg_matrices[0]);
      break;
    }
    case MultigridCoarseGridSolver::ChebyshevNonsymmetricOperator:
    {
      initialize_chebyshev_smoother_nonsymmetric_operator_coarse_grid(*mg_matrices[0]);
      break;
    }
    case MultigridCoarseGridSolver::PCG_NoPreconditioner:
    {
      // do nothing
      break;
    }
    case MultigridCoarseGridSolver::PCG_PointJacobi:
    {
      std::shared_ptr<MGCoarsePCG<Operator>> coarse_solver =
        std::dynamic_pointer_cast<MGCoarsePCG<Operator>>(mg_coarse);
      coarse_solver->update_preconditioner(*this->mg_matrices[0]);

      break;
    }
    case MultigridCoarseGridSolver::PCG_BlockJacobi:
    {
      std::shared_ptr<MGCoarsePCG<Operator>> coarse_solver =
        std::dynamic_pointer_cast<MGCoarsePCG<Operator>>(mg_coarse);
      coarse_solver->update_preconditioner(*this->mg_matrices[0]);

      break;
    }
    case MultigridCoarseGridSolver::GMRES_NoPreconditioner:
    {
      // do nothing
      break;
    }
    case MultigridCoarseGridSolver::GMRES_PointJacobi:
    {
      std::shared_ptr<MGCoarseGMRES<Operator>> coarse_solver =
        std::dynamic_pointer_cast<MGCoarseGMRES<Operator>>(mg_coarse);
      coarse_solver->update_preconditioner(*this->mg_matrices[0]);
      break;
    }
    case MultigridCoarseGridSolver::GMRES_BlockJacobi:
    {
      std::shared_ptr<MGCoarseGMRES<Operator>> coarse_solver =
        std::dynamic_pointer_cast<MGCoarseGMRES<Operator>>(mg_coarse);
      coarse_solver->update_preconditioner(*this->mg_matrices[0]);
      break;
    }
    default:
    {
      AssertThrow(false, ExcMessage("Unknown coarse-grid solver given"));
    }
  }
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::initialize_smoother(Operator &   matrix,
                                                                               unsigned int level)
{
  AssertThrow(level > 0,
              ExcMessage("Multigrid level is invalid when initializing multigrid smoother!"));

  switch(mg_data.smoother)
  {
    case MultigridSmoother::Chebyshev:
    {
      mg_smoother[level].reset(new ChebyshevSmoother<Operator, VectorTypeMG>());
      initialize_chebyshev_smoother(matrix, level);
      break;
    }
    case MultigridSmoother::ChebyshevNonsymmetricOperator:
    {
      mg_smoother[level].reset(new ChebyshevSmoother<Operator, VectorTypeMG>());
      initialize_chebyshev_smoother_nonsymmetric_operator(matrix, level);
      break;
    }
    case MultigridSmoother::GMRES:
    {
      typedef GMRESSmoother<Operator, VectorTypeMG> GMRES_SMOOTHER;
      mg_smoother[level].reset(new GMRES_SMOOTHER());

      typename GMRES_SMOOTHER::AdditionalData smoother_data;
      smoother_data.preconditioner       = mg_data.gmres_smoother_data.preconditioner;
      smoother_data.number_of_iterations = mg_data.gmres_smoother_data.number_of_iterations;

      std::shared_ptr<GMRES_SMOOTHER> smoother =
        std::dynamic_pointer_cast<GMRES_SMOOTHER>(mg_smoother[level]);
      smoother->initialize(matrix, smoother_data);
      break;
    }
    case MultigridSmoother::CG:
    {
      typedef CGSmoother<Operator, VectorTypeMG> CG_SMOOTHER;
      mg_smoother[level].reset(new CG_SMOOTHER());

      typename CG_SMOOTHER::AdditionalData smoother_data;
      smoother_data.preconditioner       = mg_data.cg_smoother_data.preconditioner;
      smoother_data.number_of_iterations = mg_data.cg_smoother_data.number_of_iterations;

      std::shared_ptr<CG_SMOOTHER> smoother =
        std::dynamic_pointer_cast<CG_SMOOTHER>(mg_smoother[level]);
      smoother->initialize(matrix, smoother_data);
      break;
    }
    case MultigridSmoother::Jacobi:
    {
      typedef JacobiSmoother<Operator, VectorTypeMG> JACOBI_SMOOTHER;
      mg_smoother[level].reset(new JACOBI_SMOOTHER());

      typename JACOBI_SMOOTHER::AdditionalData smoother_data;
      smoother_data.preconditioner = mg_data.jacobi_smoother_data.preconditioner;
      smoother_data.number_of_smoothing_steps =
        mg_data.jacobi_smoother_data.number_of_smoothing_steps;
      smoother_data.damping_factor = mg_data.jacobi_smoother_data.damping_factor;

      std::shared_ptr<JACOBI_SMOOTHER> smoother =
        std::dynamic_pointer_cast<JACOBI_SMOOTHER>(mg_smoother[level]);
      smoother->initialize(matrix, smoother_data);
      break;
    }
    default:
    {
      AssertThrow(false, ExcMessage("Specified MultigridSmoother not implemented!"));
    }
  }
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::initialize_coarse_solver(
  unsigned int const coarse_level)
{
  Operator & matrix = *mg_matrices[0];

  switch(mg_data.coarse_solver)
  {
    case MultigridCoarseGridSolver::Chebyshev:
    {
      mg_smoother[0].reset(new ChebyshevSmoother<Operator, VectorTypeMG>());
      initialize_chebyshev_smoother_coarse_grid(matrix);

      mg_coarse.reset(new MGCoarseChebyshev<VectorTypeMG, SMOOTHER>(mg_smoother[0]));
      break;
    }
    case MultigridCoarseGridSolver::ChebyshevNonsymmetricOperator:
    {
      mg_smoother[0].reset(new ChebyshevSmoother<Operator, VectorTypeMG>());
      initialize_chebyshev_smoother_nonsymmetric_operator_coarse_grid(matrix);

      mg_coarse.reset(new MGCoarseChebyshev<VectorTypeMG, SMOOTHER>(mg_smoother[0]));
      break;
    }
    case MultigridCoarseGridSolver::PCG_NoPreconditioner:
    {
      typename MGCoarsePCG<Operator>::AdditionalData additional_data;
      additional_data.preconditioner = PreconditionerCoarseGridSolver::None;

      mg_coarse.reset(new MGCoarsePCG<Operator>(matrix, additional_data));
      break;
    }
    case MultigridCoarseGridSolver::PCG_PointJacobi:
    {
      typename MGCoarsePCG<Operator>::AdditionalData additional_data;
      additional_data.preconditioner = PreconditionerCoarseGridSolver::PointJacobi;

      mg_coarse.reset(new MGCoarsePCG<Operator>(matrix, additional_data));
      break;
    }
    case MultigridCoarseGridSolver::PCG_BlockJacobi:
    {
      typename MGCoarsePCG<Operator>::AdditionalData additional_data;
      additional_data.preconditioner = PreconditionerCoarseGridSolver::BlockJacobi;

      mg_coarse.reset(new MGCoarsePCG<Operator>(matrix, additional_data));
      break;
    }
    case MultigridCoarseGridSolver::GMRES_NoPreconditioner:
    {
      typename MGCoarseGMRES<Operator>::AdditionalData additional_data;
      additional_data.preconditioner = PreconditionerCoarseGridSolver::None;

      mg_coarse.reset(new MGCoarseGMRES<Operator>(matrix, additional_data));
      break;
    }
    case MultigridCoarseGridSolver::GMRES_PointJacobi:
    {
      typename MGCoarseGMRES<Operator>::AdditionalData additional_data;
      additional_data.preconditioner = PreconditionerCoarseGridSolver::PointJacobi;

      mg_coarse.reset(new MGCoarseGMRES<Operator>(matrix, additional_data));
      break;
    }
    case MultigridCoarseGridSolver::GMRES_BlockJacobi:
    {
      typename MGCoarseGMRES<Operator>::AdditionalData additional_data;
      additional_data.preconditioner = PreconditionerCoarseGridSolver::BlockJacobi;

      mg_coarse.reset(new MGCoarseGMRES<Operator>(matrix, additional_data));
      break;
    }
    case MultigridCoarseGridSolver::AMG_ML:
    {
      mg_coarse.reset(
        new MGCoarseML<Operator, Number>(matrix, true, coarse_level, this->mg_data.coarse_ml_data));
      return;
    }
    default:
    {
      AssertThrow(false, ExcMessage("Unknown coarse-grid solver specified."));
    }
  }
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::initialize_multigrid_preconditioner()
{
  this->multigrid_preconditioner.reset(
    new MultigridPreconditioner<VectorTypeMG, Operator, MG_TRANSFER, SMOOTHER>(
      this->mg_matrices, *this->mg_coarse, this->mg_transfer, this->mg_smoother));
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::initialize_chebyshev_smoother(
  Operator &   matrix,
  unsigned int level)
{
  typedef ChebyshevSmoother<Operator, VectorTypeMG> CHEBYSHEV_SMOOTHER;
  typename CHEBYSHEV_SMOOTHER::AdditionalData       smoother_data;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  matrix.initialize_dof_vector(smoother_data.matrix_diagonal_inverse);
  matrix.calculate_inverse_diagonal(smoother_data.matrix_diagonal_inverse);
#pragma GCC diagnostic pop

  /*
  std::pair<double,double> eigenvalues = compute_eigenvalues(mg_matrices[level],
  smoother_data.matrix_diagonal_inverse);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::cout << "Eigenvalues on level l = " << level << std::endl;
    std::cout << std::scientific << std::setprecision(3)
              <<"Max EV = " << eigenvalues.second << " : Min EV = " <<
  eigenvalues.first << std::endl;
  }
  */

  smoother_data.smoothing_range     = mg_data.chebyshev_smoother_data.smoother_smoothing_range;
  smoother_data.degree              = mg_data.chebyshev_smoother_data.smoother_poly_degree;
  smoother_data.eig_cg_n_iterations = mg_data.chebyshev_smoother_data.eig_cg_n_iterations;

  std::shared_ptr<CHEBYSHEV_SMOOTHER> smoother =
    std::dynamic_pointer_cast<CHEBYSHEV_SMOOTHER>(mg_smoother[level]);
  smoother->initialize(matrix, smoother_data);
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::
  initialize_chebyshev_smoother_coarse_grid(Operator & matrix)
{
  // use Chebyshev smoother of high degree to solve the coarse grid problem approximately
  typedef ChebyshevSmoother<Operator, VectorTypeMG> CHEBYSHEV_SMOOTHER;
  typename CHEBYSHEV_SMOOTHER::AdditionalData       smoother_data;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  matrix.initialize_dof_vector(smoother_data.matrix_diagonal_inverse);
  matrix.calculate_inverse_diagonal(smoother_data.matrix_diagonal_inverse);
#pragma GCC diagnostic pop

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  std::pair<double, double> eigenvalues =
    compute_eigenvalues(matrix, smoother_data.matrix_diagonal_inverse);
#pragma GCC diagnostic pop

  double const factor = 1.1;

  smoother_data.max_eigenvalue  = factor * eigenvalues.second;
  smoother_data.smoothing_range = eigenvalues.second / eigenvalues.first * factor;

  double sigma = (1. - std::sqrt(1. / smoother_data.smoothing_range)) /
                 (1. + std::sqrt(1. / smoother_data.smoothing_range));

  double const eps = 1.e-3;

  smoother_data.degree = std::log(1. / eps + std::sqrt(1. / eps / eps - 1.)) / std::log(1. / sigma);
  smoother_data.eig_cg_n_iterations = 0;

  std::shared_ptr<CHEBYSHEV_SMOOTHER> smoother =
    std::dynamic_pointer_cast<CHEBYSHEV_SMOOTHER>(mg_smoother[0]);
  smoother->initialize(matrix, smoother_data);
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::
  initialize_chebyshev_smoother_nonsymmetric_operator(Operator & matrix, unsigned int level)
{
  typedef ChebyshevSmoother<Operator, VectorTypeMG> CHEBYSHEV_SMOOTHER;
  typename CHEBYSHEV_SMOOTHER::AdditionalData       smoother_data;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  matrix.initialize_dof_vector(smoother_data.matrix_diagonal_inverse);
  matrix.calculate_inverse_diagonal(smoother_data.matrix_diagonal_inverse);
#pragma GCC diagnostic pop

  /*
  std::pair<double,double> eigenvalues =
  compute_eigenvalues_gmres(mg_matrices[level],
  smoother_data.matrix_diagonal_inverse);
  std::cout<<"Max EW = "<< eigenvalues.second <<" : Min EW =
  "<<eigenvalues.first<<std::endl;
  */

  // use gmres to calculate eigenvalues for nonsymmetric problem
  unsigned int const eig_n_iter = 20;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  std::pair<std::complex<double>, std::complex<double>> eigenvalues =
    compute_eigenvalues_gmres(matrix, smoother_data.matrix_diagonal_inverse, eig_n_iter);
#pragma GCC diagnostic pop

  double const factor = 1.1;

  smoother_data.max_eigenvalue      = factor * std::abs(eigenvalues.second);
  smoother_data.smoothing_range     = mg_data.chebyshev_smoother_data.smoother_smoothing_range;
  smoother_data.degree              = mg_data.chebyshev_smoother_data.smoother_poly_degree;
  smoother_data.eig_cg_n_iterations = 0;

  std::shared_ptr<CHEBYSHEV_SMOOTHER> smoother =
    std::dynamic_pointer_cast<CHEBYSHEV_SMOOTHER>(mg_smoother[level]);
  smoother->initialize(matrix, smoother_data);
}

template<int dim, typename Number, typename MultigridNumber>
void
MultigridPreconditionerBase<dim, Number, MultigridNumber>::
  initialize_chebyshev_smoother_nonsymmetric_operator_coarse_grid(Operator & matrix)
{
  // use Chebyshev smoother of high degree to solve the coarse grid problem approximately
  typedef ChebyshevSmoother<Operator, VectorTypeMG> CHEBYSHEV_SMOOTHER;
  typename CHEBYSHEV_SMOOTHER::AdditionalData       smoother_data;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  matrix.initialize_dof_vector(smoother_data.matrix_diagonal_inverse);
  matrix.calculate_inverse_diagonal(smoother_data.matrix_diagonal_inverse);
#pragma GCC diagnostic pop

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  std::pair<std::complex<double>, std::complex<double>> eigenvalues =
    compute_eigenvalues_gmres(matrix, smoother_data.matrix_diagonal_inverse);
#pragma GCC diagnostic pop

  double const factor = 1.1;

  smoother_data.max_eigenvalue = factor * std::abs(eigenvalues.second);
  smoother_data.smoothing_range =
    factor * std::abs(eigenvalues.second) / std::abs(eigenvalues.first);

  double sigma = (1. - std::sqrt(1. / smoother_data.smoothing_range)) /
                 (1. + std::sqrt(1. / smoother_data.smoothing_range));

  double const eps = 1e-3;

  smoother_data.degree = std::log(1. / eps + std::sqrt(1. / eps / eps - 1)) / std::log(1. / sigma);
  smoother_data.eig_cg_n_iterations = 0;

  std::shared_ptr<CHEBYSHEV_SMOOTHER> smoother =
    std::dynamic_pointer_cast<CHEBYSHEV_SMOOTHER>(mg_smoother[0]);
  smoother->initialize(matrix, smoother_data);
}

#include "multigrid_preconditioner_base.hpp"