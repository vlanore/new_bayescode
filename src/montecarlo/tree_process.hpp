
#include "tree/tree_factory.hpp"

struct tree_process_methods  {

    template<class Tree, class Process, class CondLArray, class Params, class... Keys>
    static void backward_branch_propagate(const Tree& tree, int node, Process& process, CondLArray& old_condls, CondLArray& young_condls, const Params& params, std::tuple<Keys...>)   {

        auto timeframe = get<time_frame_field>(process);

        node_distrib_t<Process>::backward_propagate(
                young_condls[node], old_condls[node],
                timeframe(tree.parent(node)) - timeframe(node),
                get<Keys>(params)()...);

    }

    template<class Tree, class Process, class CondLArray>
    static auto recursive_backward(const Tree& tree, int node, Process& process, CondLArray& old_condls, CondLArray& young_condls)    {

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

    template<class Tree, class Process, class CondLArray, class Gen>
    static auto recursive_forward(const Tree& tree, int node, Process& process, CondLArray& young_condls, Gen& gen)    {

        node_conditional_draw(tree, node, process, young_condls,
            get<params>(process), param_keys_t<node_distrib_t<Process>>(), gen);

        for (auto c : tree.children(node))  {
            recursive_forward(tree, c, process, young_condls, gen);
        }
    }

    template<class Process, class Gen>
    static auto conditional_draw(Process& process, Gen& gen)  {

        using Distrib = node_distrib_t<Process>;
        auto initcondl = Distrib::make_init_condl();
        std::vector<typename Distrib::CondL> old_condls(process.size(), initcondl);
        std::vector<typename Distrib::CondL> young_condls(process.size(), initcondl);
        auto& tree = get<tree_field>(process);

        recursive_backward(tree, tree.root(), process, old_condls, young_condls);
        recursive_forward(tree, tree.root(), process, old_condls, young_condls, gen);
    }

    // proposals, importance sampling

    template<class Process>
    class tree_process_conditional_sampler    {

        using Distrib = node_distrib_t<Process>;
        using T = typename Distrib::T;

        Process& process;
        std::vector<typename Distrib::CondL> young_condls;

        public:

        tree_process_conditional_sampler(Process& in_process) :
                    process(in_process), 
                    young_condls(process.size(), node_distrib_t<Process>::make_init_condl())    {

            gather();
        }

        void gather()  {

            auto& tree = get<tree_field>(process);

            std::vector<typename node_distrib_t<Process>::CondL> old_condls(process.size(), 
                    node_distrib_t<Process>::make_init_condl());

            tree_process_methods::recursive_backward(tree, tree.root(),
                process, old_condls, young_condls);
        }

        template<class Gen>
        void sample_root(typename node_distrib_t<Process>::T& val, double& weight, Gen& gen)  {
            auto& tree = get<tree_field>(process);
            auto& v = get<value>(process);
            node_conditional_draw(tree, tree.root(), 
                process, young_condls,
                get<params>(process), param_keys_t<node_distrib_t<Process>>());
            val = v[tree.root()];
            weight = 1.0;
        }

        template<class Gen>
        void sample_non_root(int node, typename node_distrib_t<Process>::T& val, const typename node_distrib_t<Process>::T& parent_val, double weight, Gen& gen)    {
            auto& tree = get<tree_field>(process);
            auto& v = get<value>(process);
            v[tree.parent(node)] = parent_val;
            node_conditional_draw(tree, node, 
                process, young_condls,
                get<params>(process), param_keys_t<node_distrib_t<Process>>());
            val = v[node];
        }
    };

    template<class Process>
    static auto make_tree_process_conditional_sampler(Process& process) {
        return tree_process_conditional_sampler<Process>(process);
    }

    template<class Process, class Proposal, class BranchUpdate, class BranchLogProb>
    class tree_prior_importance_sampler  {

        const Tree& tree;
        Process& process;
        Proposal& proposal;
        BranchUpdate update;
        BranchLogProb logprob;

        tree_prior_importance_sampler(Process& in_process, Proposal& in_proposal,
            BranchUpdate in_update, BranchLogProb in_logprob) :
                process(in_process), proposal(in_proposal),
                update(in_update), logprob(in_logprob) {}

        ~tree_prior_importance_sampler() {}

        template<class Gen>
        void sample_root(typename node_distrib_t<Process>::T& val, double& weight, Gen& gen) {
            proposal.sample_root(val, weight, gen);
        }

        template<class Gen>
        void sample_non_root(int node, typename node_distrib_t<Process>::T& val, const typename node_distrib_t<Process>::T& parent_val, double& weight, Gen& gen) {
            proposal.sample_non_root(node, val, parent_val);
            update(tree.get_branch(node));
            weight += logprob(tree.get_branch(node));
        }
    };

    template<class Process, class Proposal, class Update, class LogProb>
    static auto make_tree_prior_importance_sampler(Process& process, Proposal& proposal, Update update, LogProb logprob)   {
        return tree_prior_importance_sampler<Process, Proposal, Update, LogProb>(process, proposal, update, logprob);
    }

    // particle filters

    template<class Process, class WDist>
    class tree_pf   {
        
        Process& process;
        const Tree& tree;
        WDist& wdist;
        std::vector<std::vector<typename node_distrib_t<Process>::T>> swarm;
        std::vector<double> weights;
        std::vector<int> node_ordering;
        std::vector<std::vector<int>> ancestors;
        size_t counter;

        tree_pf(Process& in_process, WDist& in_wdist, size_t n) :
            process(in_process),
            tree(get<tree_field>(process)),
            wdist(in_wdist),
            swarm(tree.nb_nodes(), 
                    std::vector<typename node_distrib_t<Process>::T>(n, typename node_distrib_t<Process>::initT())),
            weights(n, 1.0),
            node_ordering(tree.nb_nodes(), 0),
            ancestors(tree.nb_nodes(), std::vector<int>(n,0)),
            counter(0) {}

        ~tree_pf() {}

        void init() {
            counter = 0;
        }

        void run()  {

            init();

            forward_pf(tree.root());

            // choose random particle and pull out
            int c = 0;
            for (size_t i=tree.nb_nodes()-1; i>=0; i--) {
                int node = node_ordering[i];
                get<value>(process)[node] = swarm[node][c];
                c = ancestors[node][c];
            }
            return weights[c];
        }

        size_t size() {return weights.size();}

        void forward_pf(int node)   {

            node_ordering[counter] = node;
            counter++;

            if (tree.is_root(node)) {
                for (size_t i=0; i<size(); i++)  {
                    wdist.sample_root(swarm[node][i], weights[i]);
                }
            }
            else    {
                for (size_t i=0; i<size(); i++)  {
                    wdist.sample_non_root(node, 
                            swarm[node][i], swarm[tree.parent(node)][i], weights[i]);
                }
            }

            // bootstrap
            bootstrap_pf();

            // send recursion
            for (auto c : tree.children(node))  {
                forward_pf(c);
            }
        }

        void bootstrap_pf()  {
            // compute bar W
            // normalize probability vector
            // iid multinomial drawing: change states accordingly and set ancestors[i] to chosen index
        }
    };

    template<class Process, class WDist>
        static auto make_tree_pf(Process& process, WDist& wdist)    {
            return tree_pf<Process, WDist>(process, wdist);
        }
};


