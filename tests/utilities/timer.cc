
// C++
#include <fstream>
#include <iostream>
#include <sstream>

// deal.ii
#include <deal.II/base/timer.h>

#include "utilities/timings_hierarchical.h"

using namespace dealii;

double
get_fluctuation()
{
  return double(std::rand()) / double(RAND_MAX) * 0.1;
}

void
test1()
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  // clang-format off
  pcout << std::endl << std::endl<< std::endl
        << "_____________________________________________________________"<< std::endl
        << "                                                             "<< std::endl
        << "                  Timer: test 1 (basic)                      "<< std::endl
        << "_____________________________________________________________"<< std::endl
        << std::endl;
  // clang-format on

  Timer timer;
  timer.restart();

  ExaDG::TimerTree tree;

  for(unsigned int i = 0; i < 1000; ++i)
  {
    tree.insert({"General"}, 20.0 * (1 + get_fluctuation()));

    tree.insert({"General", "Part 1"}, 2.0 * (1 + get_fluctuation()));

    tree.insert({"General", "Part 2"}, 3.0 * (1 + get_fluctuation()));

    tree.insert({"General", "Part 2", "Sub a"}, 0.75 * (1 + get_fluctuation()));

    tree.insert({"General", "Part 2", "Sub b"}, 0.9 * (1 + get_fluctuation()));

    tree.insert({"General", "Part 3"}, 4.0 * (1 + get_fluctuation()));

    tree.insert({"General", "Part 3", "Sub a"}, 0.5 * (1 + get_fluctuation()));

    tree.insert({"General", "Part 3", "Sub a", "sub-sub a"}, 0.04 * (1 + get_fluctuation()));

    tree.insert({"General", "Part 3", "Sub b"}, 0.98765 * (1 + get_fluctuation()));

    tree.insert({"General"}, 2.0 * (1 + get_fluctuation()));
  }


  double wall_time = timer.wall_time();

  if(false)
  {
    pcout << "Wall time for filling the tree = " << std::scientific << wall_time << std::endl
          << std::endl;
    pcout << "Wall time for filling one item of the tree = " << std::scientific << wall_time / 10000
          << std::endl
          << std::endl;
  }

  pcout << std::endl << "timings for level = 0:" << std::endl;
  tree.print_level(pcout, 0);
  pcout << std::endl << "timings for level = 1:" << std::endl;
  tree.print_level(pcout, 1);
  pcout << std::endl << "timings for level = 2:" << std::endl;
  tree.print_level(pcout, 2);
  pcout << std::endl << "timings all:" << std::endl;
  tree.print_plain(pcout);
}

void
test2()
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  // clang-format off
  pcout << std::endl << std::endl<< std::endl
        << "_____________________________________________________________"<< std::endl
        << "                                                             "<< std::endl
        << "                  Timer: test 2 (modular coupling)           "<< std::endl
        << "_____________________________________________________________"<< std::endl
        << std::endl;
  // clang-format on

  Timer timer;
  timer.restart();

  ExaDG::TimerTree tree;
  tree.insert({"FSI"}, 100.);

  std::shared_ptr<ExaDG::TimerTree> tree_fluid;
  tree_fluid.reset(new ExaDG::TimerTree());
  tree_fluid->insert({"Fluid"}, 70.);
  tree_fluid->insert({"Fluid", "Pressure poisson"}, 40.);
  tree_fluid->insert({"Fluid", "Postprocessing"}, 10.);
  tree_fluid->insert({"Fluid", "ALE update"}, 15.);


  std::shared_ptr<ExaDG::TimerTree> tree_structure;
  tree_structure.reset(new ExaDG::TimerTree());
  tree_structure->insert({"Structure", "Right-hand side"}, 2.);
  tree_structure->insert({"Structure", "Assemble"}, 9.);
  tree_structure->insert({"Structure", "Solve"}, 14.);
  tree_structure->insert({"Structure"}, 25.);

  tree.insert({"FSI"}, tree_fluid);
  tree.insert({"FSI"}, tree_structure, "Structure");

  pcout << std::endl << "timings for level = 0:" << std::endl;
  tree.print_level(pcout, 0);
  pcout << std::endl << "timings for level = 1:" << std::endl;
  tree.print_level(pcout, 1);
  pcout << std::endl << "timings for level = 2:" << std::endl;
  tree.print_level(pcout, 2);
  pcout << std::endl << "timings all:" << std::endl;
  tree.print_plain(pcout);

  tree.clear();

  // should be empty after clear()
  pcout << std::endl << "timings for level = 0:" << std::endl;
  tree.print_level(pcout, 0);
  pcout << std::endl << "timings for level = 1:" << std::endl;
  tree.print_level(pcout, 1);
  pcout << std::endl << "timings for level = 2:" << std::endl;
  tree.print_level(pcout, 2);
  pcout << std::endl << "timings all:" << std::endl;
  tree.print_plain(pcout);

  // clear() must no touch sub-trees
  pcout << std::endl << "timings for level = 0:" << std::endl;
  tree_structure->print_level(pcout, 0);
  pcout << std::endl << "timings for level = 1:" << std::endl;
  tree_structure->print_level(pcout, 1);
  pcout << std::endl << "timings for level = 2:" << std::endl;
  tree_structure->print_level(pcout, 2);
  pcout << std::endl << "timings all:" << std::endl;
  tree_structure->print_plain(pcout);
}

int
main(int argc, char ** argv)
{
  try
  {
    Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

    deallog.depth_console(0);

    test1();

    test2();
  }
  catch(std::exception & exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }
  catch(...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }

  return 0;
}
