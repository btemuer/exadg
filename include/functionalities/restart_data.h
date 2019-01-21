/*
 * restart_data.h
 *
 *  Created on: Nov 13, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_FUNCTIONALITIES_RESTART_DATA_H_
#define INCLUDE_FUNCTIONALITIES_RESTART_DATA_H_

#include "deal.II/base/conditional_ostream.h"

#include "print_functions.h"

#include <limits>

struct RestartData
{
  RestartData()
    : write_restart(false),
      interval_time(std::numeric_limits<double>::max()),
      interval_wall_time(std::numeric_limits<double>::max()),
      interval_time_steps(std::numeric_limits<unsigned int>::max()),
      filename("restart"),
      counter(1)
  {
  }

  void
  print(ConditionalOStream & pcout)
  {
    pcout << "  Restart:" << std::endl;
    print_parameter(pcout, "Write restart", write_restart);

    if(write_restart == true)
    {
      print_parameter(pcout, "Interval physical time", interval_time);
      print_parameter(pcout, "Interval wall time", interval_wall_time);
      print_parameter(pcout, "Interval time steps", interval_time_steps);
      print_parameter(pcout, "Filename", filename);
    }
  }

  bool
  do_restart(double const       wall_time,
             double const       time,
             unsigned int const time_step_number,
             bool const         reset_counter) const
  {
    // After a restart, the counter is reset to 1. If the restart is controlled by
    // the variable interval_time, we have to reinitialize the counter because the variable time
    // is time-start_time which does not necessarily start with zero after a restart. Otherwise,
    // we would repeat all the restarts that have been written before. There is nothing to do
    // if the restart is controlled by the wall time or the time_step_number because these
    // variables are reinitialized after a restart anyway.
    if(reset_counter)
      counter += int((time + 1.e-10) / interval_time);

    bool do_restart = wall_time > interval_wall_time * counter || time > interval_time * counter ||
                      time_step_number > interval_time_steps * counter;

    if(do_restart)
      ++counter;

    return do_restart;
  }

  bool write_restart;

  // physical time
  double interval_time;

  // wall time in seconds (= hours * 3600)
  double interval_wall_time;

  // number of time steps after which to write restart
  unsigned int interval_time_steps;

  // filename for restart files
  std::string filename;

  // counter needed do decide when to write restart
  mutable unsigned int counter;
};


#endif /* INCLUDE_FUNCTIONALITIES_RESTART_DATA_H_ */
