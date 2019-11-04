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
    auto model = globom_common::make(gen);
    auto slave_m =
        slave_only_ptr([&model, &data, &gen]() { return globom_slave::make(model, *data, gen); });

    // Shared omega suffstat between models
    auto omega_ss = ss_factory::make_suffstat<OmegaPathSuffStat>([&slave_m](auto& omss) {
        omss.AddSuffStat(
            get<codon_submatrix, value>(*slave_m), get<path_suffstats>(*slave_m).get());
    });

    // communication between processes
    auto reduce_omega_ss = reduce(omega_ss->get().beta, omega_ss->get().count);
    auto omega_broadcast = broadcast(get<global_omega, omega, value>(model));

    // Draw omega @master and broadcast it
    draw(get<global_omega, omega>(model), gen);
    master_to_slave(omega_broadcast);

    // move success stats
    MoveStatsRegistry ms;

    // move schedule
    auto scheduler = make_move_scheduler([&]() {
        // move phyloprocess
        if (!master) {
            globom_slave::touch_matrices(*slave_m);
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
                omega_ss->gather();
            }
            slave_to_master(reduce_omega_ss);
            // Move omega
            if (master) { omega_sm::gibbs_resample(global_omega_(model), *omega_ss, gen); }
            master_to_slave(omega_broadcast);

            // Update slave dnode using new omega value
            if (!master) {
                // get<global_omega, omega, value>(*slave_m) = get<global_omega, omega,
                // value>(model);
                gather(get<codon_submatrix>(*slave_m));
            }
        }
    });

    // trace
    auto trace = make_custom_tracer(cmd.chain_name() + to_string(rank) + ".trace",
        trace_entry("omega", get<global_omega, omega>(model))
        // trace_entry("nucrates", get<nuc_rates, eq_freq>(*slave_m))  //
    );

    // initializing components
    ChainDriver chain_driver{cmd.chain_name(), args.every.getValue(), args.until.getValue()};

    ConsoleLogger console_logger;
    ModelTracer chain(model, cmd.chain_name() + to_string(rank) + ".chain");

    // registering components to chain driver
    chain_driver.add(scheduler);
    chain_driver.add(console_logger);
    chain_driver.add(chain);
    chain_driver.add(trace);
    chain_driver.add(ms);

    // launching chain!
    chain_driver.go();

    return 0;
}

int main(int argc, char** argv) { mpi_run(argc, argv, compute); }