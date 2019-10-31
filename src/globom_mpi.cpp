#include "mpi_components/broadcast.hpp"
#include "mpi_components/gather.hpp"
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
    auto gen = make_generator();

    // model
    auto model = slave_only_ptr([&]() { return globom::make(*data, gen); });

    // move success stats
    MoveStatsRegistry ms;

    if (!master) {
        // move schedule
        auto scheduler = make_move_scheduler([&gen, &model]() {
            // move phyloprocess
            globom::touch_matrices(*model);
            phyloprocess_(*model).Move(1.0);

            // move omega
            for (int rep = 0; rep < 30; rep++) {
                // move branch lengths
                bl_suffstats_(*model).gather();
                branchlengths_sm::gibbs_resample(
                    branch_lengths_(*model), bl_suffstats_(*model), gen);

                // move omega
                path_suffstats_(*model).gather();
                omegapath_suffstats_(*model).gather();
                omega_sm::gibbs_resample(global_omega_(*model), omegapath_suffstats_(*model), gen);

                // move nuc rates
                nucpath_suffstats_(*model).gather();
                nucrates_sm::move_nucrates(
                    nuc_rates_(*model), nucpath_suffstats_(*model), gen, 1, 1.0);
            }
        });

        // trace
        int youpi = 2;
        auto trace = make_custom_tracer(cmd.chain_name() + ".trace",  //
            trace_entry("a", [&youpi]() { return youpi; }),           //
            trace_entry("b", get<global_omega, omega>(*model))        //
        );

        // initializing components
        ChainDriver chain_driver{cmd.chain_name(), args.every.getValue(), args.until.getValue()};

        ConsoleLogger console_logger;
        ModelTracer chain(*model, cmd.chain_name() + to_string(rank) + ".chain");

        // registering components to chain driver
        chain_driver.add(scheduler);
        chain_driver.add(console_logger);
        chain_driver.add(chain);
        chain_driver.add(trace);
        chain_driver.add(ms);

        // launching chain!
        chain_driver.go();
    }
    return 0;
}

int main(int argc, char** argv) { mpi_run(argc, argv, compute); }