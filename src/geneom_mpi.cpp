#include "mpi_components/broadcast.hpp"
#include "mpi_components/gather.hpp"
#include "mpi_components/reduce.hpp"
#include "mpi_components/unique_ptr_utils.hpp"

#include "submodels/geneom_mpi_model.hpp"
#include "submodels/submodel_external_interface.hpp"

#include "traceable_collection.hpp"

using namespace std;

int compute(int argc, char* argv[]) {
    // parsing command-line arguments
    auto rank = MPI::p->rank;  // getting MPI rank
    auto size = MPI::p->size;  // ... and size
    const auto master = !rank;

    ChainCmdLine cmd{argc, argv, "SingleOmega", ' ', "0.1"};
    InferenceAppArgParse args(cmd);
    cmd.parse();

    // input data
    std::string datafile = args.alignment.getValue();
    std::string treefile = args.treefile.getValue();

    // read gene list and alignment size
    // make partition using greedy allocation based on gene size
    auto full_gene_set = MultiGeneList(args.alignment.getValue());
    Partition partition(full_gene_set.genename, full_gene_set.geneweight, size - 1, 1);

    int ngene = full_gene_set.genename.size();
    if (master) {
        MPI::p->message("total number of genes");
        MPI::p->message(ngene);
    }

    // random generator
    auto gen = make_generator(42);

    // load sequence alignments (only for slaves)
    auto data = multi_gene_data::make(args.alignment.getValue(), args.treefile.getValue(), partition);

    // Model declarations
    auto master_m = 
        master_only_ptr([&ngene, &gen] () { return geneom_master::make(ngene, gen); });
    auto slave_m =
        slave_only_ptr([&data, &gen]() { return geneom_slave::make(data, gen); });

    // communication between processes

    // reducing the suff stats for distribution of omega's across genes
    GammaSuffStat& omega_gamma_ss = master ? omega_gamma_suffstats_(*master_m).get() : omega_gamma_suffstats_(*slave_m).get();
    auto reduce_omega_gamma_ss = reduce(omega_gamma_ss.sum, omega_gamma_ss.sumlog, omega_gamma_ss.n);

    // broadcasting omega hyper params
    double& ommean_ref = master ? get<omega_hypermean, value>(*master_m) : get<omega_hypermean>(*slave_m);
    double& ominvshape_ref = master ? get<omega_hyperinvshape, value>(*master_m) : get<omega_hyperinvshape>(*slave_m);
    auto omega_hyper_broadcast = broadcast(ommean_ref, ominvshape_ref);

    // collecting omega's across genes (for tracing at the master level)
    auto& omegas = master ? get<omega_array, value>(*master_m) : get<omega_array, value>(*slave_m);
    auto omega_array_gather = gather(partition, omegas);

    master_to_slave(omega_hyper_broadcast);

    // move success stats
    MoveStatsRegistry ms;

    // move schedule
    auto scheduler = make_move_scheduler([&]() {

        MPI::p->message("move");
        if (!master) {
            geneom_slave::gene_resample_sub(*slave_m, gen);
        }

        for (int rep = 0; rep < 3; rep++) {

            if (!master) {
                geneom_slave::gene_move_params(*slave_m, gen);
                geneom_slave::gene_collect_suffstat(*slave_m);
            }

            slave_to_master(reduce_omega_gamma_ss);

            if (master) {
                geneom_master::move_hyper(*master_m, gen);
            }

            master_to_slave(omega_hyper_broadcast);
            if (!master) {
                geneom_slave::gene_update_matrices(*slave_m);
            }
        }

        if (! master)   {
            geneom_slave::gene_collect_omegas(*slave_m);
        }
        slave_to_master(omega_array_gather);

        MPI::p->message("move ok");
    });

    // initializing components
    ChainDriver chain_driver{cmd.chain_name(), args.every.getValue(), args.until.getValue()};

    ConsoleLogger console_logger;

    // registering components to chain driver
    chain_driver.add(scheduler);
    chain_driver.add(console_logger);

    if (master) {
        ModelTracer chain(*master_m, cmd.chain_name() + to_string(rank) + ".chain");
        chain_driver.add(chain);

        auto trace = make_custom_tracer(cmd.chain_name() + to_string(rank) + ".trace",
            trace_entry("mean", get<omega_hypermean>(*master_m))
        );
        chain_driver.add(trace);

        auto omegas_trace = make_custom_tracer(cmd.chain_name() + to_string(rank) + ".om",
            trace_entry("om", get<omega_array>(*master_m))
        );
        chain_driver.add(omegas_trace);
        chain_driver.add(ms);
        chain_driver.go();
    }
    else    {
        ModelTracer chain(*slave_m, cmd.chain_name() + to_string(rank) + ".chain");
        chain_driver.add(chain);

        auto trace = make_custom_tracer(cmd.chain_name() + to_string(rank) + ".trace",
            trace_entry("nomean", [] () {return 0;})
        );
        chain_driver.add(trace);
        chain_driver.add(ms);
        chain_driver.go();
    }
    return 0;
}

int main(int argc, char** argv) { mpi_run(argc, argv, compute); }
