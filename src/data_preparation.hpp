#pragma once
#include <fstream>
#include "bayes_utils/src/logging.hpp"
#include "lib/CodonSequenceAlignment.hpp"
#include "tree/implem.hpp"
#include "mpi_components/partition.hpp"

struct PreparedData {
    NHXParser parser;
    std::unique_ptr<const Tree> tree;
    FileSequenceAlignment nuc_align;
    CodonSequenceAlignment alignment;
    TaxonSet taxon_set;

    PreparedData(std::istream& alignfile, std::istream& treefile)
    // PreparedData(std::string alignfile, std::ifstream& treefile)
        : parser(treefile),
          tree(make_from_parser(parser)),
          nuc_align(alignfile),
          alignment(&nuc_align),
          taxon_set(*alignment.GetTaxonSet()) {
        // various checks and debug
        assert(tree->nb_nodes() > 0);
        DEBUG("Parsed tree with {} nodes.", tree->nb_nodes());

        assert(nuc_align.GetNtaxa() > 0 && nuc_align.GetNsite() > 0);
        DEBUG("Parsed alignment with {} sequences of length {}. Example taxon name: {}.",
            nuc_align.GetNtaxa(), nuc_align.GetNsite(), nuc_align.GetTaxonSet()->GetTaxon(0));

        assert(alignment.GetNtaxa() > 0 && alignment.GetNsite() > 0);
        DEBUG("Converted alignment to codons (new length: {}).", alignment.GetNsite());

        DEBUG("Got a taxon set of length {}. Example taxon name: {}.", taxon_set.GetNtaxa(),
            taxon_set.GetTaxon(0));
    }
};

PreparedData prepare_data(std::istream& align_stream, std::string treefile) {
    std::ifstream tree_stream{treefile};
    return {align_stream, tree_stream};
}

PreparedData prepare_data(std::string alignfile, std::string treefile) {
    std::ifstream align_stream{alignfile};
    std::ifstream tree_stream{treefile};
    return {align_stream, tree_stream};
}

auto prepare_data_ptr(std::istream& align_stream, std::string treefile) {
    std::ifstream tree_stream{treefile};
    return std::make_unique<PreparedData>(align_stream, tree_stream);
}

auto prepare_data_ptr(std::string alignfile, std::string treefile) {
    std::ifstream tree_stream{treefile};
    std::ifstream align_stream{alignfile};
    return std::make_unique<PreparedData>(align_stream, tree_stream);
}

// detailed parsing of concatenated file of gene alignments
struct multi_gene_data {

    static auto make(std::string datafile, std::string treefile, const Partition& partition)    {

        std::vector<std::unique_ptr<PreparedData>> data;

        std::ifstream is(datafile.c_str());
        std::string tmp;
        is >> tmp;
        size_t Ngene;
        is >> Ngene;

        for (size_t gene = 0; gene < Ngene; gene++) {
            string gene_name;
            is >> gene_name;

            if (partition.contains(gene_name))   {
                MPI::p->message("partition contains " + gene_name);
                auto d = prepare_data_ptr(is, treefile);
                MPI::p->message("prepare data ok");
                data.push_back(std::move(d));
                MPI::p->message("push back ok");
            }
            else    {
                // skip this alignment
                size_t ntaxa, nsite;
                is >> ntaxa >> nsite;
                std::string tmp;
                for (size_t i=0; i<ntaxa; i++)  {
                    is >> tmp >> tmp;
                }
            }
        }
        return data;
    }

    static auto make(std::string datafile, std::string treefile)    {

        std::vector<std::unique_ptr<PreparedData>> data;

        std::ifstream is(datafile.c_str());
        std::string tmp;
        is >> tmp;
        size_t Ngene;
        is >> Ngene;

        for (size_t gene = 0; gene < Ngene; gene++) {
            string gene_name;
            is >> gene_name;
            auto d = prepare_data_ptr(is, treefile);
            data.push_back(std::move(d));
        }
        return data;
    }
};

// fast reading of concatenated file of gene alignments
struct MultiGeneList {

    size_t Ngene;
    std::vector<std::string> genename;
    std::vector<int> genesize;
    std::vector<int> geneweight;

    MultiGeneList(std::string datafile) {

        std::ifstream is(datafile.c_str());
        std::string tmp;
        is >> tmp;
        is >> Ngene;

        genename.assign(Ngene,"NoName");
        genesize.assign(Ngene,0);
        geneweight.assign(Ngene,0);

        for (size_t gene = 0; gene < Ngene; gene++) {
            is >> genename[gene];
            size_t ntaxa, nsite;
            is >> ntaxa >> nsite;
            std::string tmp;
            for (size_t i=0; i<ntaxa; i++)  {
                is >> tmp >> tmp;
            }

            genesize[gene] = nsite / 3;
            geneweight[gene] = nsite * ntaxa;
        }
    }
};
