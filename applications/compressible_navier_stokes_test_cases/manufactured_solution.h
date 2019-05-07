/*
 * test_comp_NS.h
 *
 */

#ifndef APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEST_COMP_NS_H_
#define APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEST_COMP_NS_H_

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/base/function.h>

/**************************************************************************************/
/*                                                                                    */
/*                                 INPUT PARAMETERS                                   */
/*                                                                                    */
/**************************************************************************************/

/*
 *  This 2D test case is a quasi one-dimensional problem with periodic boundary
 *  conditions in x_2-direction. The velocity u_2 is zero. The energy is constant.
 *  The density and the velocity u_1 are a function of x_1 and time t.
 */

// set the number of space dimensions: DIMENSION = 2, 3
const unsigned int DIMENSION = 2;

// set the polynomial degree of the shape functions
const unsigned int FE_DEGREE = 6;

//number of quadrature points in 1D
const unsigned int QPOINTS_CONV = FE_DEGREE + 1;
const unsigned int QPOINTS_VIS = QPOINTS_CONV;

// set the number of refine levels for spatial convergence tests
const unsigned int REFINE_STEPS_SPACE_MIN = 1;
const unsigned int REFINE_STEPS_SPACE_MAX = 1;

// set the number of refine levels for temporal convergence tests
const unsigned int REFINE_STEPS_TIME_MIN = 0;
const unsigned int REFINE_STEPS_TIME_MAX = 0;

const double DYN_VISCOSITY = 0.1;
const double GAMMA = 1.4;
const double LAMBDA = 0.0;
const double R = 1.0;
const double U_0 = 1.0;
const double V_0 = 0.0;
const double RHO_0 = 1.0;
const double E_0 = 1.0e5;
const double EPSILON = 0.1*RHO_0;
const double T_MAX = 140.0;

std::string OUTPUT_FOLDER = "output_comp_ns/";
std::string FILENAME = "test";

enum class SolutionType {
  Polynomial,
  SineAndPolynomial
};

const SolutionType SOLUTION_TYPE = SolutionType::Polynomial; // SineAndPolynomial

template<int dim>
void CompNS::InputParameters<dim>::set_input_parameters()
{
  // MATHEMATICAL MODEL
  equation_type = EquationType::NavierStokes;
  right_hand_side = true;

  // PHYSICAL QUANTITIES
  start_time = 0.0;
  end_time = 7.5e-1;
  dynamic_viscosity = DYN_VISCOSITY;
  reference_density = RHO_0;
  heat_capacity_ratio = GAMMA;
  thermal_conductivity = LAMBDA;
  specific_gas_constant = R;
  max_temperature = T_MAX;

  // TEMPORAL DISCRETIZATION
  temporal_discretization = TemporalDiscretization::ExplRK3Stage7Reg2; //ExplRK4Stage8Reg2; //ExplRK3Stage4Reg2C; //ExplRK;
  order_time_integrator = 4;
  stages = 8;
  calculation_of_time_step_size = TimeStepCalculation::CFLAndDiffusion;
  time_step_size = 1.0e-3;
  max_velocity = std::sqrt(U_0*U_0+V_0*V_0);
  cfl_number = 0.025;
  diffusion_number = 0.1;
  exponent_fe_degree_cfl = 2.0;
  exponent_fe_degree_viscous = 4.0;

  // SPATIAL DISCRETIZATION

  // triangulation
  triangulation_type = TriangulationType::Distributed;

  degree = FE_DEGREE;
  degree_mapping = FE_DEGREE;
  n_q_points_conv = QPOINTS_CONV;
  n_q_points_vis = QPOINTS_VIS;

  // viscous term
  IP_factor = 1.0;

  // SOLVER

  // NUMERICAL PARAMETERS
  detect_instabilities = false;
  use_combined_operator = false;

  // OUTPUT AND POSTPROCESSING
  print_input_parameters = true;
  calculate_velocity = true;
  output_data.write_output = false;
  output_data.write_pressure = true;
  output_data.write_velocity = true;
  output_data.write_temperature = true;
  output_data.write_vorticity = true;
  output_data.write_divergence = true;
  output_data.output_folder = OUTPUT_FOLDER;
  output_data.output_name = FILENAME;
  output_data.output_start_time = start_time;
  output_data.output_interval_time = (end_time-start_time)/20;
  output_data.degree = FE_DEGREE;

  error_data.analytical_solution_available = true;
  error_data.error_calc_start_time = start_time;
  error_data.error_calc_interval_time = output_data.output_interval_time;

  // output of solver information
  solver_info_data.print_to_screen = true;
  solver_info_data.interval_time = (end_time-start_time)/10;

  // restart
  restart_data.write_restart = false;
  restart_data.interval_time = 0.5;
  restart_data.filename = OUTPUT_FOLDER + FILENAME + "_restart";
}

/**************************************************************************************/
/*                                                                                    */
/*                        GENERATE GRID AND SET BOUNDARY INDICATORS                   */
/*                                                                                    */
/**************************************************************************************/

template<int dim>
void create_grid_and_set_boundary_ids(
  std::shared_ptr<parallel::Triangulation<dim>>            triangulation,
  unsigned int const                                       n_refine_space,
  std::vector<GridTools::PeriodicFacePair<typename
    Triangulation<dim>::cell_iterator> >                   &periodic_faces)
{
  // hypercube: line in 1D, square in 2D, etc., hypercube volume is [left,right]^dim
  const double left = -1.0 , right = 0.5;
  GridGenerator::hyper_cube(*triangulation,left,right);

  typename Triangulation<dim>::cell_iterator cell = triangulation->begin(), endc = triangulation->end();
  for(;cell!=endc;++cell)
  {
    for(unsigned int face_number=0; face_number < GeometryInfo<dim>::faces_per_cell; ++face_number)
    {
      if (std::fabs(cell->face(face_number)->center()(1) -left) < 1e-12)
      {
        cell->face(face_number)->set_boundary_id(0+10);
      }
      else if (std::fabs(cell->face(face_number)->center()(1) -right) < 1e-12)
      {
        cell->face(face_number)->set_boundary_id(1+10);
      }
    }
  }

  auto tria = dynamic_cast<Triangulation<dim>*>(&*triangulation);
  GridTools::collect_periodic_faces(*tria, 0+10, 1+10, 1, periodic_faces);
  triangulation->add_periodicity(periodic_faces);

  triangulation->refine_global(n_refine_space);
}

/**************************************************************************************/
/*                                                                                    */
/*    FUNCTIONS (ANALYTICAL SOLUTION, BOUNDARY CONDITIONS, VELOCITY FIELD, etc.)      */
/*                                                                                    */
/**************************************************************************************/

/*
 *  Analytical solutions for initial field functions
 */

template<int dim>
class Solution : public Function<dim>
{
public:
  Solution (const unsigned int  n_components = dim + 2,
            const double        time = 0.)
    :
  Function<dim>(n_components, time)
  {}

  virtual ~Solution(){};

  virtual double value (const Point<dim>   &p,
                        const unsigned int component = 0) const;
};

template<int dim>
double Solution<dim>::value(const Point<dim>    &p,
                            const unsigned int  component ) const
{
  double t = this->get_time();
  const double pi = numbers::PI;

  double result = 0.0;

  double sin_pit = sin(pi * t);
  double cos_pit = cos(pi * t);
  double sin_pix = sin(pi * p[0]);
  double x3 =  p[0] * p[0] * p[0];

  double rho = 0.0;

  if(SOLUTION_TYPE == SolutionType::Polynomial)
  {
    rho = RHO_0 + EPSILON * x3 * sin_pit;
  }
  else if (SOLUTION_TYPE == SolutionType::SineAndPolynomial)
  {
    rho = RHO_0 + EPSILON * sin_pix * sin_pit;
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  if(component==0)
    result = rho;
  else if (component==1)
    result = rho * U_0 * x3 * sin_pit;
  else if (component==2)
    result = rho * V_0 * x3 * cos_pit;
  else if (component==1+dim)
    result = rho * E_0;

  return result;
}


/*
 *  Right-hand side
 */

 template<int dim>
 class RightHandSideDensity : public Function<dim>
 {
 public:
   RightHandSideDensity (const unsigned int n_components = 1,
                         const double       time = 0.)
     :
     Function<dim>(n_components, time)
   {}

   virtual ~RightHandSideDensity(){};

   virtual double value (const Point<dim>    &p,
                         const unsigned int  component = 0) const;
 };

 template<int dim>
 double RightHandSideDensity<dim>::value(const Point<dim>     &p,
                                         const unsigned int   /* component */) const
 {
   double t = this->get_time();
   double pi = numbers::PI;
   double sin_pix =sin(pi*p[0]);
   double cos_pix =cos(pi*p[0]);
   double sin_pit = sin(pi * t);
   double cos_pit = cos(pi * t);
   double x2 = p[0] * p[0];
   double x3 = p[0] * p[0] * p[0];
   double x5 = x2 * x3;

   double result = 0.0;

   if(SOLUTION_TYPE == SolutionType::Polynomial)
   {
     result = EPSILON * pi * x3 * cos_pit // = d(rho)/dt
              + 3.0 * RHO_0 * U_0 * x2 * sin_pit + 6.0 * EPSILON * U_0 * x5 * sin_pit * sin_pit; //d(rho*u1)/dx1
   }
   else if (SOLUTION_TYPE == SolutionType::SineAndPolynomial)
   {
     result = + EPSILON * pi * sin_pix * cos_pit // = d(rho)/dt
              + EPSILON * U_0 * pi * x3 * cos_pix * sin_pit * sin_pit // = u1 * d(rho)/dx1
              + (RHO_0 + EPSILON * sin_pix * sin_pit) * 3.0 * U_0 * x2 * sin_pit;// rho * d(u1)/dx1
   }
   else
   {
     AssertThrow(false, ExcMessage("Not implemented."));
   }

   return result;
 }


 template<int dim>
 class RightHandSideVelocity : public Function<dim>
 {
 public:
   RightHandSideVelocity (const unsigned int   n_components = dim,
                          const double         time = 0.)
     :
     Function<dim>(n_components, time)
   {}

   virtual ~RightHandSideVelocity(){};

   virtual double value (const Point<dim>    &p,
                         const unsigned int  component = 0) const;
 };

 template<int dim>
 double RightHandSideVelocity<dim>::value(const Point<dim>   &p,
                                          const unsigned int component) const
 {
   double t = this->get_time();
   const double pi = numbers::PI;
   const double sin_pix = sin(pi*p[0]);
   const double cos_pix = cos(pi*p[0]);
   const double sin_pit = sin(pi*t);
   const double cos_pit = cos(pi*t);
   const double p1 = p[0];
   const double p2 = p[0]*p[0];
   const double p3 = p[0]*p[0]*p[0];
   const double p5 = p[0]*p[0]*p[0]*p[0]*p[0];
   const double p6 = p5*p[0];
   const double p8 = p5*p3;

   double result = 0.0;

   if(SOLUTION_TYPE == SolutionType::Polynomial)
   {
     if(component==0)
     {
     result = + RHO_0  * pi * U_0 * p3 * cos_pit + 2.0 * U_0 * EPSILON * pi * p6 * cos_pit * sin_pit // = d(rho u1)/dt
              + (1.0-(GAMMA-1.0)/2.0) * (RHO_0 * 6.0 * U_0*U_0 * p5 * sin_pit*sin_pit + 9.0 * EPSILON * U_0*U_0  * p8  * sin_pit*sin_pit*sin_pit) // =(1 + (gamma-1)/2) d(rho u1^2)/dx1
              + (GAMMA-1.0) * E_0 * 3.0 * EPSILON * p2 * sin_pit // = (gamma-1) d(rhoE)/dx1
              - 8.0 * DYN_VISCOSITY * U_0  * p1 * sin_pit; // viscous term (= - 4/3 mu d²(u1)/dx1²)
     }
   }
   else if (SOLUTION_TYPE == SolutionType::SineAndPolynomial)
   {
     if(component==0)
     {
       result = + U_0 * EPSILON * pi * sin_pix * p3 * cos_pit * sin_pit + (RHO_0 + EPSILON * sin_pix * sin_pit) * pi * U_0 * p3 * cos_pit // = d(rho u1)/dt
                + (1.0-(GAMMA-1.0)/2.0) * ( RHO_0 * U_0*U_0 * 6.0 * p5 * sin_pit*sin_pit + U_0*U_0 * EPSILON * sin_pit*sin_pit*sin_pit * ( 6.0 * sin_pix * p5 + pi * p6 * cos_pix )) // =(1 + (gamma-1)/2) d(rho u1^2)/dx1
                + (GAMMA-1.0) * E_0 * EPSILON * pi * cos_pix * sin_pit // = (gamma-1) d(rhoE)/dx1
                - 8.0 * DYN_VISCOSITY * U_0 * p1 * sin_pit; // viscous term (= - 4/3 mu d²(u1)/dx1²)
     }
   }
   else
   {
     AssertThrow(false, ExcMessage("Not implemented."));
   }

   return result;
 }

 template<int dim>
 class RightHandSideEnergy : public Function<dim>
 {
 public:
   RightHandSideEnergy (const unsigned int   n_components = 1,
                        const double         time = 0.)
     :
     Function<dim>(n_components, time)
   {}

   virtual ~RightHandSideEnergy(){};

   virtual double value (const Point<dim>    &p,
                         const unsigned int  component = 0) const;
 };

 template<int dim>
 double RightHandSideEnergy<dim>::value(const Point<dim>   &p,
                                        const unsigned int /* component */) const
{
   double t = this->get_time();
   const double pi = numbers::PI;
   const double sin_pix = sin(pi*p[0]);
   const double cos_pix = cos(pi*p[0]);
   const double sin_pit = sin(pi*t);
   const double cos_pit = cos(pi*t);
   const double p2 = p[0]*p[0];
   const double p3 = p2*p[0];
   const double p4 = p2*p2;
   const double p5 = p2*p3;
   const double p8 = p3*p3*p2;
   const double p9 = p4*p5;
   const double p11 = p8*p3;
   const double dyn_viscosity = DYN_VISCOSITY;

   double result = 0.0;

   if(SOLUTION_TYPE == SolutionType::Polynomial)
   {
     result = + E_0 * EPSILON * pi * p3 * cos_pit // = d(rho*E)/dt
              + GAMMA * (3.0 * RHO_0 * E_0 * U_0 * p2 * sin_pit + 6.0 * EPSILON * U_0 * E_0 * p5 * sin_pit*sin_pit) // = d(rho gamma E u1)/dx1
              - (GAMMA-1.0)/2.0 * (9.0 * RHO_0 * U_0*U_0*U_0 * p8 * sin_pit*sin_pit*sin_pit + 12.0 * EPSILON * U_0*U_0*U_0 * p11 * sin_pit*sin_pit*sin_pit*sin_pit)// = -(gamma-1)/2 d(rho u1^3)/dx1
              - dyn_viscosity * 20.0 * U_0*U_0 * p4 * sin_pit*sin_pit; // viscous term (= d(u1*tau11)/dx1
   }
   else if (SOLUTION_TYPE == SolutionType::SineAndPolynomial)
   {
     result = + E_0 * EPSILON * sin_pix * pi * cos_pit // = d(rho*E)/dt
              + GAMMA * (3.0 * RHO_0 * E_0 * U_0 * p2 * sin_pit + EPSILON * E_0 * U_0 * sin_pit*sin_pit * ( 3.0 * p2 * sin_pix + pi * p3 * cos_pix)) // = d(rho gamma E u1)/dx1
              - (GAMMA-1.0)/2.0 * (9.0 * RHO_0 * U_0*U_0*U_0 * p8 * sin_pit*sin_pit*sin_pit + EPSILON * U_0*U_0*U_0 * sin_pit*sin_pit*sin_pit*sin_pit * (9.0 * p8 * sin_pix + pi * p9 * cos_pix))// = -(gamma-1)/2 d(rho u1^3)/dx1
              - dyn_viscosity * 20.0 * U_0*U_0 * p4 * sin_pit*sin_pit; // viscous term (= d(u1*tau11)/dx1
   }
   else
   {
     AssertThrow(false, ExcMessage("Not implemented."));
   }

   double f_times_u = 0;

   RightHandSideVelocity<dim> rhs_velocity(dim,t);
   Solution<dim> analytical_solution(dim+2,t);

   for(unsigned int d=0; d<dim; ++d)
   {
     f_times_u += rhs_velocity.value(p,d) * analytical_solution.value(p,1+d)/analytical_solution.value(p,0);
   }

   return result - f_times_u;
 }


 template<int dim>
 class VelocityBC : public Function<dim>
 {
 public:
   VelocityBC (const unsigned int  n_components = dim,
               const double        time = 0.)
     :
     Function<dim>(n_components, time)
   {}

   virtual ~VelocityBC(){};

   virtual double value (const Point<dim>    &p,
                         const unsigned int  component = 0) const;
 };

 template<int dim>
 double VelocityBC<dim>::value(const Point<dim>   &p,
                               const unsigned int component) const
 {
   double t = this->get_time();
   const double pi = numbers::PI;
   double x3 =  p[0] * p[0] * p[0];
   double sin_pit = sin(pi * t);
   double cos_pit = cos(pi * t);

   double result = 0.0;

   if(SOLUTION_TYPE == SolutionType::Polynomial ||
      SOLUTION_TYPE == SolutionType::SineAndPolynomial)
   {
     if (component==0)
       result = U_0 * x3 * sin_pit;
     else if (component==1)
       result = V_0 * x3 * cos_pit;
   }
   else
   {
     AssertThrow(false, ExcMessage("Not implemented."));
   }

   return result;
 }

 template<int dim>
 class EnergyBC : public Function<dim>
 {
 public:
   EnergyBC (const double time = 0.)
     :
     Function<dim>(1, time)
   {}

   virtual ~EnergyBC(){};

   virtual double value (const Point<dim>    &p,
                         const unsigned int  component = 0) const;
 };

 template<int dim>
 double EnergyBC<dim>::value(const Point<dim>   &/*p*/,
                             const unsigned int /*component*/) const
 {
   double result = 0.0;
   result = E_0;

   return result;
 }

 template<int dim>
 class DensityBC : public Function<dim>
 {
 public:
   DensityBC (const double time = 0.)
     :
     Function<dim>(1, time)
   {}

   virtual ~DensityBC(){};

   virtual double value (const Point<dim>    &p,
                         const unsigned int  component = 0) const;
 };

 template<int dim>
 double DensityBC<dim>::value(const Point<dim>   &p,
                              const unsigned int /*component*/) const
 {
   double t = this->get_time();
   const double pi = numbers::PI;
   double sin_pit = sin(pi * t);
   double sin_pix = sin(pi * p[0]);
   double x3 =  p[0] * p[0] * p[0];

   double result = 0.0;

   if(SOLUTION_TYPE == SolutionType::Polynomial)
   {
     result = RHO_0 + EPSILON * x3 * sin_pit;
   }
   else if(SOLUTION_TYPE == SolutionType::SineAndPolynomial)
   {
     result = RHO_0 + EPSILON * sin_pix * sin_pit;
   }
   else
   {
     AssertThrow(false, ExcMessage("Not implemented."));
   }

   return result;
 }

namespace CompNS
{

template<int dim>
void set_boundary_conditions(
  std::shared_ptr<CompNS::BoundaryDescriptor<dim> >        boundary_descriptor_density,
  std::shared_ptr<CompNS::BoundaryDescriptor<dim> >        boundary_descriptor_velocity,
  std::shared_ptr<CompNS::BoundaryDescriptor<dim> >        boundary_descriptor_pressure,
  std::shared_ptr<CompNS::BoundaryDescriptorEnergy<dim> >  boundary_descriptor_energy)
{
  std::shared_ptr<Function<dim> > density_bc;
  density_bc.reset(new DensityBC<dim>());
  boundary_descriptor_density->dirichlet_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(0,density_bc));

  std::shared_ptr<Function<dim> > velocity_bc;
  velocity_bc.reset(new VelocityBC<dim>());
  boundary_descriptor_velocity->dirichlet_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(0,velocity_bc));

  // do not set boundary conditions for the pressure -> neumann_bc
  std::shared_ptr<Function<dim> > pressure_bc;
  pressure_bc.reset(new Functions::ZeroFunction<dim>(1));
  boundary_descriptor_pressure->neumann_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(0,pressure_bc));

  std::shared_ptr<Function<dim> > energy_bc;
  energy_bc.reset(new EnergyBC<dim>());
  boundary_descriptor_energy->dirichlet_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(0,energy_bc));
  // set energy boundary variable
  boundary_descriptor_energy->boundary_variable.insert(std::pair<types::boundary_id,CompNS::EnergyBoundaryVariable>(0,CompNS::EnergyBoundaryVariable::Energy));
}

template<int dim>
void set_field_functions(std::shared_ptr<CompNS::FieldFunctions<dim> > field_functions)
{
  // initial solution
  std::shared_ptr<Function<dim> > initial_solution;
  initial_solution.reset(new Solution<dim>());
  field_functions->initial_solution = initial_solution;

  // rhs density
  std::shared_ptr<Function<dim> > right_hand_side_density;
  right_hand_side_density.reset(new RightHandSideDensity<dim>());
  field_functions->right_hand_side_density = right_hand_side_density;

  // rhs velocity
  std::shared_ptr<Function<dim> > right_hand_side_velocity;
  right_hand_side_velocity.reset(new RightHandSideVelocity<dim>());
  field_functions->right_hand_side_velocity = right_hand_side_velocity;

  // rhs energy
  std::shared_ptr<Function<dim> > right_hand_side_energy;
  right_hand_side_energy.reset(new RightHandSideEnergy<dim>());
  field_functions->right_hand_side_energy = right_hand_side_energy;
}

template<int dim>
void set_analytical_solution(std::shared_ptr<CompNS::AnalyticalSolution<dim> > analytical_solution)
{
  analytical_solution->solution.reset(new Solution<dim>());
}

template<int dim, typename Number>
std::shared_ptr<CompNS::PostProcessor<dim, Number> >
construct_postprocessor(CompNS::InputParameters<dim> const &param)
{
  CompNS::PostProcessorData<dim> pp_data;

  pp_data.calculate_velocity = param.calculate_velocity;
  pp_data.calculate_pressure = param.calculate_pressure;
  pp_data.output_data = param.output_data;
  pp_data.error_data = param.error_data;
  pp_data.lift_and_drag_data = param.lift_and_drag_data;
  pp_data.pressure_difference_data = param.pressure_difference_data;
  pp_data.kinetic_energy_data = param.kinetic_energy_data;
  pp_data.kinetic_energy_spectrum_data = param.kinetic_energy_spectrum_data;

  std::shared_ptr<CompNS::PostProcessor<dim, Number> > pp;
  pp.reset(new CompNS::PostProcessor<dim, Number>(pp_data));

  return pp;
}

}

#endif /* APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEST_COMP_NS_H_ */
