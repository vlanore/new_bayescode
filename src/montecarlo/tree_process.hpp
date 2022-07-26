
#include "tree/tree_factory.hpp"

struct tree_process_methods  {

    // ****************************
    // single node moves (and auxiliary functions)

    template<class Process, class Params, class... Keys>
    static double unpack_node_logprob(Process& process, int node, const Params& params, std::tuple<Keys...>)   {

        using Distrib = node_distrib_t<Process>;
        auto& tree = get<tree_field>(process);
        auto timeframe = get<time_frame_field>(process);
        auto& v = get<value>(process);
        return tree.is_root(node) ?
            Distrib::logprob(v[node], true, v[node], 0, get<Keys>(params)()...) :
            Distrib::logprob(v[node], false, v[tree.parent(node)],
                timeframe(tree.parent(node)) - timeframe(node),
                get<Keys>(params)()...);
    }

    template<class Process>
    static double node_logprob(Process& process, int node) {
        return unpack_node_logprob(process, node, 
                get<params>(process), param_keys_t<node_distrib_t<Process>>());
    }

    template<class Process>
    static double around_node_logprob(Process& process, int node) {
        double tot = node_logprob(process, node);
        for (auto c : get<tree_field>(process).children(node))  {
            tot += node_logprob(process, c);
        }
        return tot;
    }

    template<class Process>
    static auto around_node_logprob(Process& process)   {
        return [&p = process] (int node) {return around_node_logprob(p, node);};
    }

    template<class Tree, class Process, class Proposal, class BranchUpdate, class BranchLogProb, class Gen>
    static void single_node_mh_move(Tree& tree, int node, Process& process, Proposal propose, BranchUpdate update, BranchLogProb logprob, Gen& gen)  {

        auto& x = get<value>(process)[node];
        auto bk = x;
        double logprobbefore = node_logprob(process,node) + tree_factory::sum_around_node(tree, logprob)(node);
        double logh = propose(x, gen);
        tree_factory::do_around_node(tree, update)(node);
        double logprobafter = node_logprob(process,node) + tree_factory::sum_around_node(tree, logprob)(node);
        double delta = logprobafter - logprobbefore + logh;
        bool accept = decide(delta, gen);
        if (! accept)   {
            x = bk;
            tree_factory::do_around_node(tree, update)(node);
        }
    }

    template<class Tree, class Process, class Proposal, class BranchUpdate, class BranchLogProb, class Gen>
    static void recursive_mh_move(Tree& tree, int node, Process& process, Proposal propose, BranchUpdate update, BranchLogProb logprob, Gen& gen)    {
        single_node_mh_move(tree, node, process, propose, update, logprob, gen);
        for (auto c : tree.children(node)) {
            recursive_mh_move(tree, c, process, propose, update, logprob, gen);
        }
        single_node_mh_move(tree, node, process, propose, update, logprob, gen);
    }

    template<class Process, class Proposal, class BranchUpdate, class BranchLogProb, class Gen>
    static void node_by_node_mh_move(Process& process, Proposal propose, BranchUpdate update, BranchLogProb logprob, Gen& gen)  {
        auto& tree = get<tree_field>(process);
        recursive_mh_move(tree, tree.root(), process, propose, update, logprob, gen);
    }

    template<class Process, class BranchUpdate, class BranchLogProb, class Gen>
    static void node_by_node_mh_move(Process& process, double tuning, BranchUpdate update, BranchLogProb logprob, Gen& gen)  {
        auto& tree = get<tree_field>(process);
        auto propose = node_distrib_t<Process>::kernel(tuning);
        recursive_mh_move(tree, tree.root(), process, propose, update, logprob, gen);
    }

    // ****************************
    // add suffstat

    template<class ParamKey, class Tree, class Process, class SS, class Params, class... Keys>
    static void local_add_branch_suffstat(ParamKey key, Tree& tree, int node, Process& process, SS& ss, const Params& params, std::tuple<Keys...>)  {

        auto& v = get<value>(process);
        auto timeframe = get<time_frame_field>(process);

        node_distrib_t<Process>::add_branch_suffstat(
                key, ss, v[node], v[tree.parent(node)],
                timeframe(tree.parent(node)) - timeframe(node),
                get<Keys>(params)()...);
    }

    template<class ParamKey, class Tree, class Process, class SS>
    static void recursive_add_branch_suffstat(ParamKey key, Tree& tree, int node, Process& process, SS& ss) {
        if (! tree.is_root(node)) {
            local_add_branch_suffstat(key, tree, node, process, ss, 
                    get<params>(process), param_keys_t<node_distrib_t<Process>>());
        }
        for (auto c : tree.children(node)) {
            recursive_add_branch_suffstat(key, tree, c, process, ss);
        }
    }

    template<class ParamKey, class Process, class SS>
    static void add_branch_suffstat(Process& process, SS& ss)    {
        auto& tree = get<tree_field>(process);
        ParamKey key;
        recursive_add_branch_suffstat(key, tree, tree.root(), process, ss);
    }

    // ****************************
    // backward forward routines

    template<class Tree, class Process, class CondLArray, class Params, class... Keys>
    static void backward_branch_propagate(const Tree& tree, int node, Process& process, CondLArray& old_condls, CondLArray& young_condls, const Params& params, std::tuple<Keys...>)   {

        auto timeframe = get<time_frame_field>(process);

        node_distrib_t<Process>::backward_propagate(
                young_condls[node], old_condls[node],
                timeframe(tree.parent(node)) - timeframe(node),
                get<Keys>(params)()...);

    }

    template<class Tree, class Process, class CondLArray>
    static void recursive_backward(const Tree& tree, int node, Process& process, CondLArray& old_condls, CondLArray& young_condls)    {

        auto& v = get<value>(process);
        auto& clamp = get<constraint>(process);
        node_distrib_t<Process>::backward_initialize(v[node], clamp[node], young_condls[node]);

        for (auto c : tree.children(node))  {
            recursive_backward(tree, c, process, old_condls, young_condls);
            node_distrib_t<Process>::multiply_conditional_likelihood(young_condls[node], old_condls[c]);
        }

        if (! tree.is_root(node))   {
            backward_branch_propagate(tree, node, process, old_condls, young_condls,
                        get<params>(process), param_keys_t<node_distrib_t<Process>>());
        }
    }

    template<class Tree, class Process, class CondLArray, class Params, class... Keys, class Gen>
    static void node_conditional_draw(const Tree& tree, int node, Process& process, CondLArray& young_condls, const Params& params, std::tuple<Keys...>, Gen& gen)   {

        auto& v = get<value>(process);
        auto& clamp = get<constraint>(process);
        auto timeframe = get<time_frame_field>(process);

        if (tree.is_root(node)) {
            node_distrib_t<Process>::root_conditional_draw(
                    v[node], clamp[node],
                    young_condls[node],
                    get<Keys>(params)()..., gen);
        }
        else    {
            node_distrib_t<Process>::non_root_conditional_draw(
                    v[node], clamp[node], v[tree.parent(node)],
                    timeframe(tree.parent(node)) - timeframe(node),
                    young_condls[node],
                    get<Keys>(params)()..., gen);
        }
    }

    template<class Tree, class Process, class CondLArray, class Params, class... Keys>
    static double node_conditional_logprob(const Tree& tree, int node, Process& process, CondLArray& young_condls, const Params& params, std::tuple<Keys...>)   {

        auto& v = get<value>(process);
        auto timeframe = get<time_frame_field>(process);

        return tree.is_root(node) ?

            node_distrib_t<Process>::root_conditional_logprob(
                v[node], 
                young_condls[node],
                get<Keys>(params)()...):

            node_distrib_t<Process>::non_root_conditional_logprob(
                v[node], v[tree.parent(node)],
                timeframe(tree.parent(node)) - timeframe(node),
                young_condls[node],
                get<Keys>(params)()...);
    }

    template<class Tree, class Process, class CondLArray, class Gen>
    static void recursive_forward(const Tree& tree, int node, Process& process, CondLArray& young_condls, Gen& gen)    {

        node_conditional_draw(tree, node, process, young_condls,
            get<params>(process), param_keys_t<node_distrib_t<Process>>(), gen);

        for (auto c : tree.children(node))  {
            recursive_forward(tree, c, process, young_condls, gen);
        }
    }

    // ****************************
    // conditional draw (full backward-forward - not persisent)

    template<class Process, class Gen>
    static void conditional_draw(Process& process, Gen& gen)  {

        using Distrib = node_distrib_t<Process>;
        auto initcondl = Distrib::make_init_condl();
        std::vector<typename Distrib::CondL> old_condls(get<value>(process).size(), initcondl);
        std::vector<typename Distrib::CondL> young_condls(get<value>(process).size(), initcondl);
        auto& tree = get<tree_field>(process);

        recursive_backward(tree, tree.root(), process, old_condls, young_condls);
        recursive_forward(tree, tree.root(), process, young_condls, gen);
    }

    // ****************************
    // proposals, importance sampling

    template<class Process>
    class conditional_sampler    {

        Process& process;
        std::vector<typename node_distrib_t<Process>::CondL> young_condls;

        public:

        conditional_sampler(Process& in_process) :
            process(in_process), 
            young_condls(get<value>(process).size(), node_distrib_t<Process>::make_init_condl())    {
            gather();
        }

        void gather()  {

            auto& tree = get<tree_field>(process);

            std::vector<typename node_distrib_t<Process>::CondL> old_condls(
                    get<value>(process).size(), 
                    node_distrib_t<Process>::make_init_condl());

            recursive_backward(tree, tree.root(),
                process, old_condls, young_condls);
        }

        template<class Gen>
        void root_draw(typename node_distrib_t<Process>::T& val, Gen& gen)  {
            auto& tree = get<tree_field>(process);
            auto& v = get<value>(process);
            node_conditional_draw(tree, tree.root(), 
                process, young_condls,
                get<params>(process), param_keys_t<node_distrib_t<Process>>(), gen);
            val = v[tree.root()];
        }

        template<class Gen>
        void non_root_draw(int node, typename node_distrib_t<Process>::T& val, const typename node_distrib_t<Process>::T& parent_val, Gen& gen)    {
            auto& tree = get<tree_field>(process);
            auto& v = get<value>(process);
            v[tree.parent(node)] = parent_val;
            node_conditional_draw(tree, node, 
                process, young_condls,
                get<params>(process), param_keys_t<node_distrib_t<Process>>(), gen);
            val = v[node];
        }

        double root_conditional_logprob() {
            auto& tree = get<tree_field>(process);
            auto& v = get<value>(process);
            return node_conditional_logprob(tree, tree.root(), 
                process, young_condls,
                get<params>(process), param_keys_t<node_distrib_t<Process>>());
        }

        double non_root_conditional_logprob()   {
            auto& tree = get<tree_field>(process);
            auto& v = get<value>(process);
            return node_conditional_logprob(tree, node, 
                process, young_condls,
                get<params>(process), param_keys_t<node_distrib_t<Process>>());
        }
    };

    template<class Process>
    static auto make_conditional_sampler(Process& process) {
        return conditional_sampler<Process>(process);
    }

    template<class Process, class Proposal, class BranchUpdate, class BranchLogProb>
    class prior_importance_sampler  {

        Process& process;
        Proposal& proposal;
        BranchUpdate update;
        BranchLogProb logprob;

        public:

        prior_importance_sampler(Process& in_process, Proposal& in_proposal,
            BranchUpdate in_update, BranchLogProb in_logprob) :
                process(in_process), proposal(in_proposal),
                update(in_update), logprob(in_logprob) {}

        ~prior_importance_sampler() {}

        template<class Gen>
        void root_draw(typename node_distrib_t<Process>::T& val, double& log_weight, Gen& gen) {
            proposal.root_draw(val, gen);
        }

        template<class Gen>
        void non_root_draw(int node, typename node_distrib_t<Process>::T& val, const typename node_distrib_t<Process>::T& parent_val, double& log_weight, Gen& gen) {
            proposal.non_root_draw(node, val, parent_val, gen);
            update(get<tree_field>(process).get_branch(node));
            log_weight += logprob(get<tree_field>(process).get_branch(node));
        }
    };

    template<class Process, class Proposal, class Update, class LogProb>
    static auto make_prior_importance_sampler(Process& process, Proposal& proposal, Update update, LogProb logprob)   {
        return prior_importance_sampler<Process, Proposal, Update, LogProb>(process, proposal, update, logprob);
    }

    template<class Process, class Proposal, class BranchUpdate, class BranchLogProb>
    class importance_sampler  {

        Process& process;
        Proposal& proposal;
        BranchUpdate update;
        BranchLogProb logprob;

        public: 

        importance_sampler(Process& in_process, Proposal& in_proposal,
            BranchUpdate in_update, BranchLogProb in_logprob) :
                process(in_process), proposal(in_proposal),
                update(in_update), logprob(in_logprob) {}

        ~importance_sampler() {}

        template<class Gen>
        void root_draw(typename node_distrib_t<Process>::T& val, double& log_weight, Gen& gen) {
            proposal.root_draw(val, gen);
            log_weight -= proposal.root_logprob(val);
            log_weight += node_logprob(process, get<tree_field>(process).root());
        }

        template<class Gen>
        void non_root_draw(int node, typename node_distrib_t<Process>::T& val, const typename node_distrib_t<Process>::T& parent_val, double& log_weight, Gen& gen) {
            proposal.non_root_draw(node, val, parent_val, gen);
            update(get<tree_field>(process).get_branch(node));
            log_weight -= proposal.non_root_logprob(val, parent_val);
            log_weight += node_logprob(process, node);
            log_weight += logprob(get<tree_field>(process).get_branch(node));
        }
    };

    template<class Process, class Proposal, class Update, class LogProb>
    static auto make_importance_sampler(Process& process, Proposal& proposal, Update update, LogProb logprob)   {
        return importance_sampler<Process, Proposal, Update, LogProb>(process, proposal, update, logprob);
    }

    // ****************************
    // particle filters

    template<class Process, class WDist>
    class tree_pf   {
        
        Process& process;
        const Tree& tree;
        WDist& wdist;
        std::vector<std::vector<typename node_distrib_t<Process>::T>> swarm;
        std::vector<double> log_weights;
        std::vector<int> node_ordering;
        std::vector<std::vector<int>> ancestors;
        size_t counter;

        public:

        tree_pf(Process& in_process, WDist& in_wdist, size_t n) :
            process(in_process),
            tree(get<tree_field>(process)),
            wdist(in_wdist),
            swarm(tree.nb_nodes(), 
                    std::vector<typename node_distrib_t<Process>::T>(n, get<value>(process)[0])),
            log_weights(n, 0),
            node_ordering(tree.nb_nodes(), 0),
            ancestors(tree.nb_nodes(), std::vector<int>(n,0)),
            counter(0) {}

        ~tree_pf() {}

        void init() {
            counter = 0;
        }

        template<class Gen>
        double run(Gen& gen)  {

            init();

            forward_pf(tree.root(), gen);

            // choose random particle and pull out
            int c = 0;
            for (size_t i=tree.nb_nodes()-1; i>=0; i--) {
                int node = node_ordering[i];
                get<value>(process)[node] = swarm[node][c];
                c = ancestors[node][c];
            }
            return log_weights[c];
        }

        size_t size() {return log_weights.size();}

        template<class Gen>
        void forward_pf(int node, Gen& gen)   {

            node_ordering[counter] = node;
            counter++;

            if (tree.is_root(node)) {
                for (size_t i=0; i<size(); i++)  {
                    wdist.root_draw(swarm[node][i], log_weights[i], gen);
                }
            }
            else    {
                for (size_t i=0; i<size(); i++)  {
                    wdist.non_root_draw(node, 
                            swarm[node][i], swarm[tree.parent(node)][i], log_weights[i], gen);
                }
            }

            // bootstrap
            bootstrap_pf(gen);

            // send recursion
            for (auto c : tree.children(node))  {
                forward_pf(c, gen);
            }
        }

        template<class Gen>
        void bootstrap_pf(Gen& gen)  {
            // compute bar W
            // normalize probability vector
            // iid multinomial drawing: change states accordingly and set ancestors[i] to chosen index
        }
    };

    template<class Process, class WDist>
        static auto make_particle_filter(Process& process, WDist& wdist, size_t n)    {
            return tree_pf<Process, WDist>(process, wdist, n);
        }
};


