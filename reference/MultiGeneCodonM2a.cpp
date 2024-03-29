#include <cmath>
#include <fstream>
#include "MultiGeneCodonM2aModel.hpp"
#include "SlaveChainDriver.hpp"
#include "components/ChainCheckpoint.hpp"
#include "components/ChainDriver.hpp"
#include "components/ConsoleLogger.hpp"
#include "components/InferenceAppArgParse.hpp"
#include "components/StandardTracer.hpp"

using namespace std;
using std::unique_ptr;

class MultiGeneCodonM2aArgParse : public BaseArgParse {
  public:
    MultiGeneCodonM2aArgParse(ChainCmdLine& cmd) : BaseArgParse(cmd) {}

    ValueArg<string> nucrates{"", "nucrates",
        "Possible values are, shared, shrunken, independent(ind)", false, "shrunken", "string",
        cmd};
    ValueArg<string> bl{"", "blmode", "Possible values are, shared, shrunken, independent(ind)",
        false, "shrunken", "string", cmd};
    param_mode_t blmode() { return decode_mode("bl", bl.getValue()); }
    param_mode_t nucmode() { return decode_mode("nucrates", bl.getValue()); }

  private:
    param_mode_t decode_mode(std::string option_name, std::string mode) {
        if (mode == "shared") {
            return shared;
        } else if (mode == "shrunken") {
            return shrunken;
        } else if ((mode == "ind") || (mode == "independent")) {
            return independent;
        } else {
            cerr << "error: does not recognize command after -" << option_name << "\n";
            exit(1);
        }
    }
};

template <class D, class M>
struct AppData {
    unique_ptr<D> chain_driver;
    unique_ptr<M> model;
};

template <class D, class M>
AppData<D, M> load_appdata(ChainCmdLine& cmd) {
    if (cmd.resume_from_checkpoint()) {
        AppData<D, M> d;
        std::ifstream is = cmd.checkpoint_file();
        d.chain_driver = unique_ptr<D>(new D(is));
        is >> d.model;
        d.model->Update();
        return d;
    } else {
        AppData<D, M> d;
        InferenceAppArgParse app(cmd);
        MultiGeneCodonM2aArgParse args(cmd);
        cmd.parse();
        d.chain_driver =
            unique_ptr<D>(new D(cmd.chain_name(), app.every.getValue(), app.until.getValue()));
        d.model = unique_ptr<M>(new M(
            app.alignment.getValue(), app.treefile.getValue(), args.blmode(), args.nucmode()));
        d.model->Update();
        return d;
    }
}

void compute(int argc, char** argv) {
    ChainCmdLine cmd{argc, argv, "MultiGeneCodonM2a", ' ', "0.1"};

    if (!MPI::p->rank) {
        auto d = load_appdata<ChainDriver, MultiGeneCodonM2aModel>(cmd);
        ConsoleLogger console_logger;
        ChainCheckpoint chain_checkpoint(cmd.chain_name() + ".param", *d.chain_driver, *d.model);
        StandardTracer trace(*d.model, cmd.chain_name());
        d.chain_driver->add(*d.model);
        d.chain_driver->add(console_logger);
        d.chain_driver->add(chain_checkpoint);
        d.chain_driver->add(trace);
        d.chain_driver->go();
    } else {
        auto d = load_appdata<SlaveChainDriver, MultiGeneCodonM2aModel>(cmd);
        d.chain_driver->add(*d.model);
        d.chain_driver->go();
    }
}

int main(int argc, char** argv) { mpi_run(argc, argv, compute); }
