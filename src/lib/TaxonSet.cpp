#include "TaxonSet.hpp"
#include <iostream>
#include "BiologicalSequences.hpp"
#include "bayes_utils/src/logging.hpp"
#include "tree/implem.hpp"

using namespace std;

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
//     TaxonSet
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

TaxonSet::TaxonSet(const std::vector<string> &names) : Ntaxa(names.size()), taxlist(names) {
    for (int i = 0; i < Ntaxa; i++) {
        if (taxmap[names[i]] != 0) { FAIL("Found several taxa with same name: {}", names[i]); }
        taxmap[names[i]] = i + 1;
    }
}

// @Thibault: check this function
std::vector<int> TaxonSet::get_reverse_index_table(const Tree *tree) const {
    std::vector<int> ret(Ntaxa, -1);
    for (size_t node = 0; node < tree->nb_nodes(); node++) {
        if (tree->is_leaf(node)) {
            int index = GetTaxonIndex(tree->node_name(node));
            assert((index >= 0) && (index < Ntaxa));
            ret[index] = node;
        }
    }
    for (int tax=0; tax<Ntaxa; tax++)    {
        if (ret[tax] == -1) {
            std::cerr << "taxon " << taxlist[tax] << " not found in tree\n";
        }
    }
    return ret;
}

std::vector<int> TaxonSet::get_index_table(const Tree *tree) const {
    std::vector<int> ret(tree->nb_nodes(), -1);
    for (size_t node = 0; node < tree->nb_nodes(); node++) {
        if (tree->is_leaf(node)) {
            if (tree->node_name(node) == "") { FAIL("Leaf has no name"); }
            ret[node] = GetTaxonIndex(tree->node_name(node));
        }
    }
    INFO("Get index table ok");
    return ret;
}

TaxonSet::TaxonSet(const TaxonSet &from)
    : Ntaxa(from.GetNtaxa()), taxmap(from.taxmap), taxlist(from.taxlist) {}

void TaxonSet::ToStream(ostream &os) const {
    os << Ntaxa << '\n';
    for (int i = 0; i < Ntaxa; i++) { os << taxlist[i] << '\n'; }
}

int TaxonSet::GetTaxonIndexWithIncompleteName(string taxname) const {
    int found = -1;
    for (int i = 0; i < Ntaxa; i++) {
        if (taxlist[i].substr(0, taxname.length()) == taxname) {
            if (found != -1) { FAIL("Taxon found twice: {}", taxname); }
            found = i;
        }
    }
    if (found == -1) {
        for (int i = 0; i < Ntaxa; i++) {
            if (taxname.substr(0, taxlist[i].length()) == taxlist[i]) {
                if (found != -1) { FAIL("Taxon found twice: {}", taxname); }
                found = i;
            }
        }
    }
    return found;
}
