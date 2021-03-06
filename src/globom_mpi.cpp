#include "mpi_components/broadcast.hpp"
#include "mpi_components/gather.hpp"
#include "mpi_components/reduce.hpp"
#include "mpi_components/unique_ptr_utils.hpp"
#include "submodels/globom_mpi_model.hpp"
#include "submodels/submodel_external_interface.hpp"
#include "traceable_collection.hpp"

using namespace std;

int compute(int argc, char* argv[]) {
    // parsing command-line arguments
    auto rank = MPI::p->rank;  // getting MPI rank
    auto size = MPI::p->size;  // ... and size
    const auto master = !rank;

    IndexSet indices(size - 1);
    std::generate(
        indices.begin(), indices.end(), [n = 0]() mutable { return std::to_string(n++); });
    Partition partition(indices, size - 1, 1);

    ChainCmdLine cmd{argc, argv, "SingleOmega", ' ', "0.1"};
    InferenceAppArgParse args(cmd);
    cmd.parse();

    // input data
    std::unique_ptr<PreparedData> data =
        master ? nullptr : prepare_data_ptr(args.alignment.getValue(), args.treefile.getValue());

    if (!master) {
        MPI::p->message(data->tree->nb_nodes());
        MPI::p->message(data->alignment.GetNsite());
    }
    // random generator
    auto gen = make_generator(42);
    Random::InitRandom(42);

    // Model declarations
    auto master_m = 
        master_only_ptr([&gen] () { return globom_master::make(gen); });
    auto slave_m =
        slave_only_ptr([&data, &gen]() { return globom_slave::make(*data, gen); });

    // communication between processes
    OmegaPathSuffStat& omega_ss = master ? omegapath_suffstats_(*master_m).get() : omegapath_suffstats_(*slave_m).get();
    auto reduce_omega_ss = reduce(omega_ss.beta, omega_ss.count);

    double& omega_ref = master ? get<global_omega, value>(*master_m) : get<global_omega>(*slave_m);
    auto omega_broadcast = broadcast(omega_ref);

    // Draw omega @master and broadcast it
    if (master) {
        draw(get<global_omega>(*master_m), gen);
    }
    master_to_slave(omega_broadcast);

    // move success stats
    MoveStatsRegistry ms;

    // move schedule
    auto scheduler = make_move_scheduler([&]() {
        // move phyloprocess
        if (!master) {
            gather(get<nuc_rates, nuc_matrix>(*slave_m));
            gather(codon_submatrix_(*slave_m));
            phyloprocess_(*slave_m).Move(1.0);
        }

        for (int rep = 0; rep < 30; rep++) {
            if (!master) {
                // move branch lengths
                bl_suffstats_(*slave_m).gather();
                branchlengths_sm::gibbs_resample(
                    branch_lengths_(*slave_m), bl_suffstats_(*slave_m), gen);
                // move nuc rates
                nucpath_suffstats_(*slave_m).gather();
                nucrates_sm::move_nucrates(
                    nuc_rates_(*slave_m), nucpath_suffstats_(*slave_m), gen, 1, 1.0);

                // gather omega suffstats
                path_suffstats_(*slave_m).gather();
                omegapath_suffstats_(*slave_m).gather();
            }
            slave_to_master(reduce_omega_ss);
            // Move omega
            if (master) {
                gibbs_resample(global_omega_(*master_m), omegapath_suffstats_(*master_m), gen);
            }
            master_to_slave(omega_broadcast);

            // Update slave dnode using new omega value
            if (!master) {
                gather(get<codon_submatrix>(*slave_m));
            }
        }
    });

    // initializing components
    ChainDriver chain_driver{cmd.chain_name(), args.every.getValue(), args.until.getValue()};

    ConsoleLogger console_logger;
    // not clear how to deal with master/slave issues here

    // registering components to chain driver
    chain_driver.add(scheduler);
    chain_driver.add(console_logger);

    // not clear how to deal with that
    /*
    ModelTracer chain(*master_m, cmd.chain_name() + to_string(rank) + ".chain");
    chain_driver.add(chain);

    auto trace = make_custom_tracer(cmd.chain_name() + to_string(rank) + ".trace",
        trace_entry("omega", get<global_omega, omega>(*master_m))
    );
    chain_driver.add(trace);
    */

    chain_driver.add(ms);

    // launching chain!
    chain_driver.go();

    return 0;
}

int main(int argc, char** argv) { mpi_run(argc, argv, compute); }
