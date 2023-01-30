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

#ifndef INCLUDE_EXADG_DARCY_USER_INTERFACE_ENUM_TYPES_H_
#define INCLUDE_EXADG_DARCY_USER_INTERFACE_ENUM_TYPES_H_

#include <string>

namespace ExaDG
{
namespace Darcy
{
/* ************************************************************************************ */
/*                                                                                      */
/*                                 MATHEMATICAL MODEL                                   */
/*                                                                                      */
/* ************************************************************************************ */

/*
 *  ProblemType refers to the underlying physics of the flow problem and describes
 *  whether the considered flow problem is expected to be a steady or unsteady solution
 *  of the Darcy equations.
 */
enum class ProblemType
{
  Undefined,
  Steady,
  Unsteady
};

std::string
enum_to_string(ProblemType enum_type);

enum class MeshMovementType
{
  Function,
  Elasticity
};

std::string
enum_to_string(MeshMovementType const enum_type);


/* ************************************************************************************ */
/*                                                                                      */
/*                                 PHYSICAL QUANTITIES                                  */
/*                                                                                      */
/* ************************************************************************************ */

// there are currently no enums for this section


/* ************************************************************************************ */
/*                                                                                      */
/*                             TEMPORAL DISCRETIZATION                                  */
/*                                                                                      */
/* ************************************************************************************ */

/*
 *  SolverType refers to the numerical solution of the incompressible Navier-Stokes
 *  equations and describes whether a steady or an unsteady solver is used.
 *  For the Darcy equations, the problem type and the solver type are expected to match.
 */
enum class TemporalSolverType
{
  Undefined,
  Steady,
  Unsteady
};

/*
 *  Temporal discretization method
 */
enum class TemporalDiscretizationMethod
{
  Undefined,
  BDFCoupled
};

std::string
enum_to_string(TemporalDiscretizationMethod enum_type);

/*
 * Calculation of time step size
 */
enum class TimeStepCalculation
{
  Undefined,
  UserSpecified
};

std::string
enum_to_string(TimeStepCalculation enum_type);


/* ************************************************************************************ */
/*                                                                                      */
/*                              SPATIAL DISCRETIZATION                                  */
/*                                                                                      */
/* ************************************************************************************ */

/*
 *  Spatial discretization method.
 */
enum class SpatialDiscretizationMethod
{
  L2
};

std::string
enum_to_string(SpatialDiscretizationMethod enum_type);

/*
 *  Polynomial degree of pressure shape functions in relation to velocity degree
 */
enum class DegreePressure
{
  MixedOrder,
  EqualOrder
};

std::string
enum_to_string(DegreePressure enum_type);

/*
 *  Type of imposition of Dirichlet BC's:
 *
 *  direct: u⁺ = g
 *  mirror: u⁺ = -u⁻ + 2g
 *
 */
enum class TypeDirichletBCs
{
  Direct,
  Mirror
};

std::string
enum_to_string(TypeDirichletBCs enum_type);

/*
 * Different options for adjusting the pressure level in case of pure Dirichlet
 * boundary conditions
 */
enum class AdjustPressureLevel
{
  ApplyZeroMeanValue
};

std::string
enum_to_string(AdjustPressureLevel enum_type);

/* ************************************************************************************ */
/*                                                                                      */
/*                                COUPLED DARCY SOLVER                                  */
/*                                                                                      */
/* ************************************************************************************ */

/*
 * Solver for the Darcy problem
 *
 * - use GMRES as default.
 *
 * - FGMRES might be necessary if a Krylov method is used inside the preconditioner
 *   (e.g., as multigrid smoother or as multigrid coarse grid solver).
 */
enum class LinearSolverMethod
{
  GMRES,
  FGMRES
};

std::string
enum_to_string(LinearSolverMethod enum_type);

/*
 *  Preconditioner type for linearized Navier-Stokes problem
 *
 *  - use BlockTriangular as default (typically best option in terms of time-to-solution, i.e.
 *    BlockDiagonal needs significantly more iterations and BlockTriangularFactorization reduces
 *    number of iterations only slightly but is significantly more expensive)
 */
enum class PreconditionerCoupled
{
  None,
  BlockDiagonal,
  BlockTriangular,
  BlockTriangularFactorization
};

std::string
enum_to_string(PreconditionerCoupled enum_type);

/*
 *  preconditioner for velocity/momentum operator
 *
 *  steady problems:
 *
 *  - use Multigrid as default
 *
 *  unsteady problems:
 *
 *  - use InverseMassMatrix as default. As a rule of thumb, only try other
 *    preconditioners if the number of iterations is significantly larger than 10.
 */
enum class VelocityBlockPreconditioner
{
  None,
  PointJacobi,
  BlockJacobi,
  InverseMomentumMatrix
};

std::string
enum_to_string(VelocityBlockPreconditioner enum_type);

enum class SchurComplementPreconditioner
{
  None,
  LaplaceOperator
};

std::string
enum_to_string(SchurComplementPreconditioner enum_type);

} // namespace Darcy
} // namespace ExaDG
#endif /* INCLUDE_EXADG_DARCY_USER_INTERFACE_ENUM_TYPES_H_ */
