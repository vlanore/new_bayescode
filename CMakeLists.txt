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
  )
add_library (bayescode_lib STATIC ${BAYESCODE_LIB})

set(BASE_LIBS
    bayescode_lib
    tree_lib
    global_lib
)

# Single omega (globom)
add_executable(globom "src/globom.cpp")
target_link_libraries(globom ${BASE_LIBS})

# Site-specific omega (siteom)
add_executable(siteom "src/siteom.cpp")
target_link_libraries(siteom ${BASE_LIBS})

# Mixture-model omega (mixom)
add_executable(mixom "src/mixom.cpp")
target_link_libraries(mixom ${BASE_LIBS})

# tests
add_executable(all_tests "src/all_tests.cpp")
target_link_libraries(all_tests ${BASE_LIBS} ${MPI_LIBRARIES})

add_executable(tree_test "src/tree/test.cpp")
target_link_libraries(tree_test tree_lib)

add_executable(mpi_par_test "src/mpi_components/test/mpi_test.cpp")
target_link_libraries(mpi_par_test ${MPI_LIBRARIES})

add_executable(mpi_example "src/mpi_example.cpp")
target_link_libraries(mpi_example ${MPI_LIBRARIES})

add_executable(simu_and_infer "examples/simu_and_infer_example.cpp")
target_link_libraries(simu_and_infer ${BASE_LIBS})

add_executable(coin_tosses_mpi "src/coin_tosses_beta_MPI.cpp")
target_link_libraries(coin_tosses_mpi ${MPI_LIBRARIES})
add_executable(coin_tosses_modular_mpi "src/coin_tosses_beta_modular_MPI.cpp")
target_link_libraries(coin_tosses_modular_mpi ${MPI_LIBRARIES})

add_executable(globomMPI "src/globom_mpi.cpp")
target_link_libraries(globomMPI ${BASE_LIBS} ${MPI_LIBRARIES})