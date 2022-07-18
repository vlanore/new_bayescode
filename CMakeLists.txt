cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(bayescode CXX)

# MPI
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

# Compilation options
option(COVERAGE_MODE "For coverage mode using g++ " OFF) #OFF by default
option(DEBUG_MODE "Debug mode (with asserts and such) " OFF) #OFF by default
set(ALWAYSON_CXX_FLAGS "--std=c++14 -Wall -Wextra -Wno-unused-parameter -Wfatal-errors ")
if(COVERAGE_MODE)
    set(CMAKE_CXX_FLAGS "-O0 -fprofile-arcs -ftest-coverage ${ALWAYSON_CXX_FLAGS}") # coverage mode
    message("-- INFO: Compiling in coverage mode.\n-- INFO: flags are: " ${CMAKE_CXX_FLAGS})
elseif(DEBUG_MODE)
    set(CMAKE_CXX_FLAGS "-O2 -g -ftemplate-backtrace-limit=0 ${ALWAYSON_CXX_FLAGS}") # debug mode
    message("-- INFO: Compiling in debug mode.\n-- INFO: flags are: " ${CMAKE_CXX_FLAGS})
else()
    set(CMAKE_CXX_FLAGS "-O3 -DNDEBUG ${ALWAYSON_CXX_FLAGS}") # release mode
    message("-- INFO: Compiling in release mode.\n-- INFO: flags are: " ${CMAKE_CXX_FLAGS})
endif(COVERAGE_MODE)

include_directories("src")
include_directories("src/lib")
include_directories("utils")
include_directories("utils/bayes_toolbox/src")
include_directories("utils/nhx-parser/src")
include_directories("utils/bayes_toolbox/utils")
include_directories("utils/bayes_toolbox/utils/tagged_tuple/utils")
include_directories("utils/bayes_toolbox/utils/tagged_tuple/utils/bayes_utils/utils")

# Static libs
add_library (tree_lib STATIC "src/tree/implem.cpp" "utils/nhx-parser/src/nhx-parser.cpp")
add_library (global_lib STATIC "src/global/Random.cpp")

set(BAYESCODE_LIB
    src/lib/AAMutSelOmegaCodonSubMatrix.cpp
    src/lib/BranchSitePath.cpp
    src/lib/CodonSequenceAlignment.cpp
    src/lib/CodonStateSpace.cpp
    src/lib/CodonSubMatrix.cpp
    src/lib/GTRSubMatrix.cpp
    src/lib/PhyloProcess.cpp
    src/lib/SequenceAlignment.cpp
    src/lib/StateSpace.cpp
    src/lib/SubMatrix.cpp
    src/lib/TaxonSet.cpp
    src/lib/linalg.cpp
  )
add_library (bayescode_lib STATIC ${BAYESCODE_LIB})

set(BASE_LIBS
    bayescode_lib
    tree_lib
    global_lib
)

# brownian_clock_globom (brownian_clock_globom)
add_executable(brownian_clock_globom "src/brownian_clock_globom.cpp")
target_link_libraries(brownian_clock_globom ${BASE_LIBS})

# mapping_brownian_clock_globom (mapping_brownian_clock_globom)
add_executable(mapping_brownian_clock_globom "src/mapping_brownian_clock_globom.cpp")
target_link_libraries(mapping_brownian_clock_globom ${BASE_LIBS})

# mutseldp (mutseldp)
add_executable(mutseldp "src/mutseldp.cpp")
target_link_libraries(mutseldp ${BASE_LIBS})

# selac (selac)
add_executable(selac "src/selac.cpp")
target_link_libraries(selac ${BASE_LIBS})

# selacNe (selacNe)
add_executable(selacNe "src/selacNe.cpp")
target_link_libraries(selacNe ${BASE_LIBS})

# coevol (coevol)
add_executable(coevol "src/coevol.cpp")
target_link_libraries(coevol ${BASE_LIBS})

# coevol (multicoevol)
add_executable(multicoevol "src/multigene_mpi_coevol.cpp")
target_link_libraries(multicoevol ${BASE_LIBS} ${MPI_LIBRARIES})

# Single omega (globom)
add_executable(globom "src/globom.cpp")
target_link_libraries(globom ${BASE_LIBS})

# geneom MPI
add_executable(geneom_mpi "src/geneom_mpi.cpp")
target_link_libraries(geneom_mpi ${BASE_LIBS} ${MPI_LIBRARIES})

# tests
add_executable(all_tests "src/all_tests.cpp")
target_link_libraries(all_tests ${BASE_LIBS} ${MPI_LIBRARIES})

add_executable(tree_test "src/tree/test.cpp")
target_link_libraries(tree_test tree_lib)

add_executable(mpi_par_test "src/mpi_components/test/mpi_test.cpp")
target_link_libraries(mpi_par_test ${MPI_LIBRARIES})

