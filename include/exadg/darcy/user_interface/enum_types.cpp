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

// deal.II
#include <deal.II/base/exceptions.h>

// ExaDG
#include <exadg/darcy/user_interface/enum_types.h>

namespace ExaDG
{
namespace Darcy
{
/* ************************************************************************************ */
/*                                                                                      */
/*                                 MATHEMATICAL MODEL                                   */
/*                                                                                      */
/* ************************************************************************************ */

std::string
enum_to_string(ProblemType const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case ProblemType::Undefined:
      string_type = "Undefined";
      break;
    case ProblemType::Steady:
      string_type = "Steady";
      break;
    case ProblemType::Unsteady:
      string_type = "Unsteady";
      break;
    default:
      AssertThrow(false, dealii::ExcNotImplemented());
  }

  return string_type;
}

std::string
enum_to_string(MeshMovementType const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case MeshMovementType::Function:
      string_type = "Function";
      break;
    case MeshMovementType::Elasticity:
      string_type = "Elasticity";
      break;
    default:
      AssertThrow(false, dealii::ExcNotImplemented());
  }

  return string_type;
}

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

std::string
enum_to_string(TemporalDiscretizationMethod const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case TemporalDiscretizationMethod::Undefined:
      string_type = "Undefined";
      break;
    case TemporalDiscretizationMethod::BDFCoupled:
      string_type = "BDF (coupled) solution";
      break;
    default:
      AssertThrow(false, dealii::ExcNotImplemented());
  }

  return string_type;
}

std::string
enum_to_string(TimeStepCalculation const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case TimeStepCalculation::Undefined:
      string_type = "Undefined";
      break;
    case TimeStepCalculation::UserSpecified:
      string_type = "UserSpecified";
      break;
    default:
      AssertThrow(false, dealii::ExcNotImplemented());
  }

  return string_type;
}

/* ************************************************************************************ */
/*                                                                                      */
/*                              SPATIAL DISCRETIZATION                                  */
/*                                                                                      */
/* ************************************************************************************ */


std::string
enum_to_string(SpatialDiscretizationMethod const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case SpatialDiscretizationMethod::L2:
      string_type = "L2 - Discontinuous Galerkin";
      break;
    default:
      AssertThrow(false, dealii::ExcNotImplemented());
  }

  return string_type;
}

std::string
enum_to_string(DegreePressure const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case DegreePressure::MixedOrder:
      string_type = "Mixed-order";
      break;
    case DegreePressure::EqualOrder:
      string_type = "Equal-order";
      break;
    default:
      AssertThrow(false, dealii::ExcNotImplemented());
  }

  return string_type;
}

std::string
enum_to_string(TypeDirichletBCs const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case TypeDirichletBCs::Direct:
      string_type = "Direct";
      break;
    case TypeDirichletBCs::Mirror:
      string_type = "Mirror";
      break;
    default:
      AssertThrow(false, dealii::ExcNotImplemented());
  }

  return string_type;
}

std::string
enum_to_string(AdjustPressureLevel const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case AdjustPressureLevel::ApplyZeroMeanValue:
      string_type = "ApplyZeroMeanValue";
      break;
    default:
      AssertThrow(false, dealii::ExcNotImplemented());
  }

  return string_type;
}


/* ************************************************************************************ */
/*                                                                                      */
/*                                COUPLED DARCY SOLVER                                  */
/*                                                                                      */
/* ************************************************************************************ */

std::string
enum_to_string(LinearSolverMethod const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case LinearSolverMethod::GMRES:
      string_type = "GMRES";
      break;
    case LinearSolverMethod::FGMRES:
      string_type = "FGMRES";
      break;
    default:
      AssertThrow(false, dealii::ExcNotImplemented());
  }

  return string_type;
}

std::string
enum_to_string(PreconditionerCoupled const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case PreconditionerCoupled::None:
      string_type = "None";
      break;
    case PreconditionerCoupled::BlockDiagonal:
      string_type = "BlockDiagonal";
      break;
    case PreconditionerCoupled::BlockTriangular:
      string_type = "BlockTriangular";
      break;
    case PreconditionerCoupled::BlockTriangularFactorization:
      string_type = "BlockTriangularFactorization";
      break;
    default:
      AssertThrow(false, dealii::ExcNotImplemented());
  }

  return string_type;
}

std::string
enum_to_string(VelocityBlockPreconditioner const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case VelocityBlockPreconditioner::None:
      string_type = "None";
      break;
    case VelocityBlockPreconditioner::PointJacobi:
      string_type = "PointJacobi";
      break;
    case VelocityBlockPreconditioner::BlockJacobi:
      string_type = "BlockJacobi";
      break;
    case VelocityBlockPreconditioner::InverseMomentumMatrix:
      string_type = "InverseMomentumMatrix";
      break;
    default:
      AssertThrow(false, dealii::ExcNotImplemented());
  }

  return string_type;
}

std::string
enum_to_string(SchurComplementPreconditioner const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case SchurComplementPreconditioner::None:
      string_type = "None";
      break;
    case SchurComplementPreconditioner::LaplaceOperator:
      string_type = "LaplaceOperator";
      break;
    default:
      AssertThrow(false, dealii::ExcNotImplemented());
  }

  return string_type;
}
} // namespace Darcy
} // namespace ExaDG
