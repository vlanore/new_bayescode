#include "mpi_components/broadcast.hpp"
#include "mpi_components/gather.hpp"
#include "mpi_components/reduce.hpp"
#include "mpi_components/reduce_suffstats.hpp"
#include "mpi_components/unique_ptr_utils.hpp"

#include "submodels/multigene_mpi_coevol_model.hpp"
#include "submodels/submodel_external_interface.hpp"

#include "traceable_collection.hpp"

using namespace std;

int compute(int argc, char* argv[]) {

    // random generator
    auto gen = make_generator(42);

    // parsing command-line arguments
    auto rank = MPI::p->rank;  // getting MPI rank
    auto size = MPI::p->size;  // ... and size
    const auto master = !rank;

    ChainCmdLine cmd{argc, argv, "MultiGeneCoevol", ' ', "0.1"};
    CoevolInferenceAppArgParse args(cmd);
    cmd.parse();

    std::string datafile = args.alignment.getValue();
    std::string treefile = args.treefile.getValue();
    std::string contdatafile = args.contdatafile.getValue();
    std::string rootfile = args.rootfile.getValue();

    // read gene list and alignment size
    // make partition using greedy allocation based on gene size
    auto full_gene_set = MultiGeneList(datafile);
    Partition partition(full_gene_set.genename, full_gene_set.geneweight, size - 1, 1);

    int ngene = full_gene_set.genename.size();
    if (master) {
        MPI::p->message("total number of genes");
        MPI::p->message(ngene);
    }

    // load sequence alignments (only for slaves)
    auto data = multi_gene_data_without_tree::make(datafile, partition);
    auto codon_statespace = std::make_unique<CodonStateSpace>(Universal);

    // input tree
    std::ifstream tis(treefile);
    NHXParser parser(tis);
    auto tree = make_from_parser(parser);

    // traits
    auto cont_data = FileContinuousData(contdatafile);

    // root mean and var for brownian process
    std::ifstream is(rootfile);
    size_t ncont;
    is >> ncont;
    auto root_mean = std::vector<double>(ncont,0);
    auto root_var = std::vector<double>(ncont,0);
    if (master) {
        for (size_t i=0; i<ncont; i++)  {
            is >> root_mean[i] >> root_var[i];
        }
    }

    // model declarations
    auto master_m = 
        master_only_ptr([&tree, &codon_statespace, &cont_data, &root_mean, &root_var, &gen] () { 
                return coevol_master::make(tree.get(), codon_statespace.get(), cont_data, root_mean, root_var, gen);});

    auto slave_m =
        slave_only_ptr([&tree, &data, &gen]() {
                return coevol_slave::make(tree.get(), data, gen);});

    // communication between processes

    // broadcasting global parameters master -> slave
    auto broadcast_dsom = master ?
        broadcast(get<synrate,value>(*master_m), get<omega,value>(*master_m)) :
        broadcast(get<synrate>(*slave_m), get<omega>(*slave_m));

    auto broadcast_nuc = master ?
        broadcast(get<nuc_rates,exch_rates,value>(*master_m), get<nuc_rates,eq_freq,value>(*master_m)) :
        broadcast(get<nucrr>(*slave_m), get<nucstat>(*slave_m));

    auto reduce_dsomss = master ? 
        reduce_suffstats(dsom_suffstats_(*master_m)) : 
        reduce_suffstats(dsom_suffstats_(*slave_m));
        
    // reducing path suff stats slave -> master
    auto reduce_nucss = master ? 
        reduce_suffstats(nucpath_suffstats_(*master_m)) : 
        reduce_suffstats(nucpath_suffstats_(*slave_m));

    // move success stats
    MoveStatsRegistry ms;

    // move schedule
    auto scheduler = make_move_scheduler([&]() {

        MPI::p->message("move");
        if (!master) {
            coevol_slave::gene_resample_sub(*slave_m, gen);
            coevol_slave::gene_collect_path_suffstats(*slave_m);
        }

        for (int rep = 0; rep < 30; rep++) {

            // collect ds om suffstats
            if (!master) {
                coevol_slave::gene_collect_dsomsuffstats(*slave_m);
            }
            slave_to_master(reduce_dsomss);

            // move chrono, brownian process and sigma
            if (master) {
                coevol_master::move_chrono(*master_m, gen);
                coevol_master::move_process(*master_m, gen);
                coevol_master::move_sigma(*master_m, gen);
            }

            // broadcast ds om, update gene codon matrices and collect nuc suffstats
            master_to_slave(broadcast_dsom);
            if (!master) {
                coevol_slave::update_codon_matrices(*slave_m);
                coevol_slave::gene_collect_nucsuffstats(*slave_m);
            }
            slave_to_master(reduce_nucss);

            // move nuc rates
            if (master) {
                coevol_master::move_nuc(*master_m, gen);
            }

            // broadcast nucrates and update gene matrices
            master_to_slave(broadcast_nuc);
            if (!master) {
                coevol_slave::update_nuc_matrix(*slave_m);
                coevol_slave::update_codon_matrices(*slave_m);
            }
        }

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
            trace_entry("tl", [&model=*master_m] () {return coevol_master::get_total_ds(model);}),
            trace_entry("om", [&model=*master_m] () {return coevol_master::get_mean_omega(model);}),
            trace_entry("sigma01", [&model=*master_m ] () {return get<sigma,value>(model)[0][1];}),
            trace_entry("sigma02", [&model=*master_m] () {return get<sigma,value>(model)[0][2];}),
            trace_entry("sigma12", [&model=*master_m] () {return get<sigma,value>(model)[1][2];})
        );

        chain_driver.add(trace);

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
