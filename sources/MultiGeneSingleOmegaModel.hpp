
// this is a multigene version of singleomegamodel
//
// - branch lengths are shared across genes, and are iid Exponential of rate lambda
// - nucleotide relative exchangeabilities and stationaries are also shared across genes (uniform Dirichlet)
// - the array of gene-specific omega's are iid gamma with hyperparameters omegahypermean and omegahyperinvshape
//
// the sequence of MCMC moves is as follows:
// - genes resample substitution histories, gather path suff stats and move their omega's
// - master receives the array of omega's across genes, moves their hyperparameters and then broadcast the new value of these hyperparams
// - master collects branch length suff stats across genes, moves branch lengths and broadcasts their new value
// - master collects nuc path suffstats across genes, moves nuc rates and broadcasts their new value

#include "SingleOmegaModel.hpp"
#include "Parallel.hpp"
#include "MultiGeneProbModel.hpp"

class MultiGeneSingleOmegaModel : public MultiGeneProbModel {

    private:

	Tree* tree;
	CodonSequenceAlignment* refcodondata;
	const TaxonSet* taxonset;

    string treefile;

	int Ntaxa;
	int Nbranch;

    // branch lengths are shared across genes
    // iid expo (gamma of shape 1 and scale lambda)
    // where lambda is a hyperparameter
	double lambda;
	BranchIIDGamma* branchlength;
	
    // nucleotide rates are shared across genes
	vector<double> nucrelrate;
	vector<double> nucstat;
	GTRSubMatrix* nucmatrix;

    // each gene has its own omega
    // omegaarray[gene], for gene=1..Ngene
    // iid gamma, with hyperparameters omegahypermean and hyperinvshape
	double omegahypermean;
	double omegahyperinvshape;
	IIDGamma* omegaarray;

    // suffstat for gene-specific omega's
    // as a function of omegahypermean and omegahyperinvshape
	GammaSuffStat omegahypersuffstat;

    // suffstats for paths, as a function of branch lengths
	PoissonSuffStatBranchArray* lengthpathsuffstatarray;
    // suff stats for branch lengths, as a function of lambda
	GammaSuffStat hyperlengthsuffstat;

    // suffstats for paths, as a function of nucleotide rates
    NucPathSuffStat nucpathsuffstat;

    // each gene defines its own SingleOmegaModel
    std::vector<SingleOmegaModel*> geneprocess;

    // total log likelihood (summed across all genes)
    double lnL;
    // total logprior for gene-specific variables (here, omega only)
    // summed over all genes
    double GeneLogPrior;

    public:

    //-------------------
    // Construction and allocation
    //-------------------

    MultiGeneSingleOmegaModel(string datafile, string intreefile, int inmyid, int innprocs) : MultiGeneProbModel(inmyid,innprocs) {

        AllocateAlignments(datafile);
        treefile = intreefile;

        refcodondata = new CodonSequenceAlignment(refdata, true);
        taxonset = refdata->GetTaxonSet();
        Ntaxa = refdata->GetNtaxa();

        // get tree from file (newick format)
        tree = new Tree(treefile);

        // check whether tree and data fits together
        tree->RegisterWith(taxonset);

        tree->SetIndices();
        Nbranch = tree->GetNbranch();

        if (! myid) {
            cerr << "number of taxa : " << Ntaxa << '\n';
            cerr << "number of branches : " << Nbranch << '\n';
            cerr << "tree and data fit together\n";
        }
    }

    void Allocate() {

        lambda = 10;
        branchlength = new BranchIIDGamma(*tree,1.0,lambda);
        lengthpathsuffstatarray = new PoissonSuffStatBranchArray(*tree);

        nucrelrate.assign(Nrr,0);
        Random::DirichletSample(nucrelrate,vector<double>(Nrr,1.0/Nrr),((double) Nrr));

        nucstat.assign(Nnuc,0);
        Random::DirichletSample(nucstat,vector<double>(Nnuc,1.0/Nnuc),((double) Nnuc));

        nucmatrix = new GTRSubMatrix(Nnuc,nucrelrate,nucstat,true);

        omegahypermean = 1.0;
        omegahyperinvshape = 1.0;
		omegaarray = new IIDGamma(GetLocalNgene(),1.0,1.0);

        lnL = 0;
        GeneLogPrior = 0;

        if (! GetMyid())    {
            geneprocess.assign(0,(SingleOmegaModel*) 0);
        }
        else    {
            geneprocess.assign(GetLocalNgene(),(SingleOmegaModel*) 0);

            for (int gene=0; gene<GetLocalNgene(); gene++)   {
                geneprocess[gene] = new SingleOmegaModel(GetLocalGeneName(gene),treefile);
                geneprocess[gene]->Allocate();
            }
        }
    }

    void FastUpdate()   {

        branchlength->SetScale(lambda);
        double alpha = 1.0 / omegahyperinvshape;
        double beta = alpha / omegahypermean;
        omegaarray->SetShape(alpha);
        omegaarray->SetScale(beta);
    }

    void MasterUpdate() override {

        FastUpdate();

        if (nprocs > 1) {
            MasterSendGlobalBranchLengths();
            MasterSendGlobalNucRates();
            MasterSendOmegaHyperParameters();
            MasterSendOmega();
            MasterReceiveLogProbs();
        }
    }

    void SlaveUpdate() override {

        SlaveReceiveGlobalBranchLengths();
        SlaveReceiveGlobalNucRates();
        SlaveReceiveOmegaHyperParameters();
        SlaveReceiveOmega();
        GeneUpdate();
        SlaveSendLogProbs();
    }

    void GeneUpdate()	{
        for (int gene=0; gene<GetLocalNgene(); gene++)   {
            geneprocess[gene]->Update();
        }
    }

    void MasterPostPred(string name) override {
        FastUpdate();
        if (nprocs > 1) {
            MasterSendGlobalBranchLengths();
            MasterSendGlobalNucRates();
            MasterSendOmegaHyperParameters();
            MasterSendOmega();
        }
    }

    void SlavePostPred(string name) override {
        SlaveReceiveGlobalBranchLengths();
        SlaveReceiveGlobalNucRates();
        SlaveReceiveOmegaHyperParameters();
        SlaveReceiveOmega();
        GenePostPred(name);
    }

    void GenePostPred(string name)	{
        for (int gene=0; gene<GetLocalNgene(); gene++)   {
            geneprocess[gene]->PostPred(GetLocalGeneName(gene) + name);
        }
    }

	CodonStateSpace* GetCodonStateSpace() const {
		return (CodonStateSpace*) refcodondata->GetStateSpace();
	}

    const vector<double>& GetOmegaArray() const {
        return omegaarray->GetArray();
    }

    //-------------------
    // Traces and Monitors
    //-------------------

    void TraceHeader(ostream& os) const {

        os << "#logprior\tlnL\tlength\t";
        os << "meanomega\t";
        os << "varomega\t";
        os << "omegahypermean\tinvshape\t";
        os << "statent\t";
        os << "rrent\n";
    }

    void Trace(ostream& os) const {
		os << GetLogPrior() << '\t';
		os << GetLogLikelihood() << '\t';
        os << branchlength->GetTotalLength() << '\t';
        os << omegaarray->GetMean() << '\t';
        os << omegaarray->GetVar() << '\t';
        os << omegahypermean << '\t' << omegahyperinvshape << '\t';
		os << Random::GetEntropy(nucstat) << '\t';
		os << Random::GetEntropy(nucrelrate) << '\n';
		os.flush();
    }

	void Monitor(ostream& os) const {}

	void MasterFromStream(istream& is) {
        is >> lambda;
        is >> *branchlength;
        is >> nucrelrate;
        is >> nucstat;
        is >> omegahypermean;
        is >> omegahyperinvshape;
        is >> *omegaarray;
    }

	void MasterToStream(ostream& os) const {
        os << lambda << '\n';
        os << *branchlength << '\n';
        os << nucrelrate << '\n';
        os << nucstat << '\n';
        os << omegahypermean << '\n';
        os << omegahyperinvshape << '\n';
        os << *omegaarray << '\n';
    }

    //-------------------
    // Updates
    //-------------------

	void TouchNucMatrix()	{
		nucmatrix->CopyStationary(nucstat);
		nucmatrix->CorruptMatrix();
	}

    void NoUpdate() {}

    //-------------------
    // Log Prior and Likelihood
    //-------------------

    double GetLogPrior() const {
		double total = 0;
		total += BranchLengthsHyperLogPrior();
		total += BranchLengthsLogPrior();
        total += NucRatesLogPrior();
        total += OmegaHyperLogPrior();
		total += OmegaLogPrior();
		return total;
    }

	double BranchLengthsHyperLogPrior() const {
		return -lambda / 10;
	}

	double BranchLengthsLogPrior() const {
		return branchlength->GetLogProb();
	}

    double OmegaHyperLogPrior() const {
        double total = 0;
        total -= omegahypermean;
        total -= omegahyperinvshape;
        return total;
    }

    double OmegaLogPrior()   const {
        return omegaarray->GetLogProb();
    }

    double NucRatesLogPrior() const {
        return 0;
    }

    double GetLogLikelihood() const {
        return lnL;
    }

    //-------------------
    // Suff Stat Log Probs
    //-------------------

    // suff stat for moving branch lengths hyperparameter (lambda)
	double BranchLengthsHyperSuffStatLogProb() const {
		return hyperlengthsuffstat.GetLogProb(1.0,lambda);
	}

    // suff stats for moving nuc rates
    double NucRatesSuffStatLogProb() const {
        return nucpathsuffstat.GetLogProb(*nucmatrix,*GetCodonStateSpace());
    }

    // suff stats for moving omega hyper parameters
    double OmegaHyperSuffStatLogProb() const {
        double alpha = 1.0 / omegahyperinvshape;
        double beta = alpha / omegahypermean;
        return omegahypersuffstat.GetLogProb(alpha,beta);
    }

    //-------------------
    // Log Probs for MH moves
    //-------------------

    // log prob for moving branch lengths hyperparameter (lambda)
    double BranchLengthsHyperLogProb() const {
        return BranchLengthsHyperLogPrior() + BranchLengthsHyperSuffStatLogProb();
    }

    // log prob for moving nuc rates
    double NucRatesLogProb() const {
        return NucRatesLogPrior() + NucRatesSuffStatLogProb();
    }

    // log prob for moving omega hyperparameters
    double OmegaHyperLogProb() const {
        return OmegaHyperLogPrior() + OmegaHyperSuffStatLogProb();
    }

    //-------------------
    // Moves
    //-------------------

    // all methods starting with Master are called only by master
    // for each such method, there is a corresponding method called by slave, and starting with Slave
    //
    // all methods starting with Gene are called only be slaves, and do some work across all genes allocated to that slave

    void MasterMove() override {

		int nrep = 30;

		for (int rep=0; rep<nrep; rep++)	{

            // collect gene-specific omega's
            MasterReceiveOmega();
            // move omegahypermean and omegahyperinvshape
            MoveOmegaHyperParameters();
            // send omega hyperparams to slaves
            MasterSendOmegaHyperParameters();

            // collect pathsuffstats (as a function of branch lengths) across genes
            MasterReceiveLengthSuffStat();
            // resample branch lengths and hyperparams
            ResampleBranchLengths();
            MoveBranchLengthsHyperParameter();
            // send branch lengths to slaves
            MasterSendGlobalBranchLengths();

            // collect nucpathsuffstat (i.e. path suff stats as a function of nuc rates) across genes
            MasterReceiveNucPathSuffStat();
            // resample nuc rates
            MoveNucRates();
            // send nucrates to slaves
            MasterSendGlobalNucRates();
        }

        MasterReceiveOmega();
        MasterReceiveLogProbs();
    }

    // slave move
    void SlaveMove() override {

        GeneResampleSub(1.0);

		int nrep = 30;

		for (int rep=0; rep<nrep; rep++)	{

            GeneCollectPathSuffStat();
            MoveGeneOmegas();
            SlaveSendOmega();
            SlaveReceiveOmegaHyperParameters();

            SlaveSendLengthSuffStat();
            SlaveReceiveGlobalBranchLengths();

            SlaveSendNucPathSuffStat();
            SlaveReceiveGlobalNucRates();
        }

        SlaveSendOmega();
        SlaveSendLogProbs();
    }

    void GeneResampleSub(double frac)  {

        for (int gene=0; gene<GetLocalNgene(); gene++)   {
            geneprocess[gene]->ResampleSub(frac);
        }
    }

    void GeneCollectPathSuffStat()  {
        for (int gene=0; gene<GetLocalNgene(); gene++)   {
            geneprocess[gene]->CollectPathSuffStat();
        }
    }

    void MoveGeneOmegas()  {
        for (int gene=0; gene<GetLocalNgene(); gene++)   {
            geneprocess[gene]->MoveOmega();
            (*omegaarray)[gene] = geneprocess[gene]->GetOmega();
        }
    }

    void ResampleBranchLengths()    {
		branchlength->GibbsResample(*lengthpathsuffstatarray);
    }

	void MoveBranchLengthsHyperParameter()	{

		hyperlengthsuffstat.Clear();
		hyperlengthsuffstat.AddSuffStat(*branchlength);

        ScalingMove(lambda,1.0,10,&MultiGeneSingleOmegaModel::BranchLengthsHyperLogProb,&MultiGeneSingleOmegaModel::NoUpdate,this);
        ScalingMove(lambda,0.3,10,&MultiGeneSingleOmegaModel::BranchLengthsHyperLogProb,&MultiGeneSingleOmegaModel::NoUpdate,this);

		branchlength->SetScale(lambda);
    }

    void MoveOmegaHyperParameters()  {

		omegahypersuffstat.Clear();
		omegahypersuffstat.AddSuffStat(*omegaarray);

        ScalingMove(omegahypermean,1.0,10,&MultiGeneSingleOmegaModel::OmegaHyperLogProb,&MultiGeneSingleOmegaModel::NoUpdate,this);
        ScalingMove(omegahypermean,0.3,10,&MultiGeneSingleOmegaModel::OmegaHyperLogProb,&MultiGeneSingleOmegaModel::NoUpdate,this);
        ScalingMove(omegahyperinvshape,1.0,10,&MultiGeneSingleOmegaModel::OmegaHyperLogProb,&MultiGeneSingleOmegaModel::NoUpdate,this);
        ScalingMove(omegahyperinvshape,0.3,10,&MultiGeneSingleOmegaModel::OmegaHyperLogProb,&MultiGeneSingleOmegaModel::NoUpdate,this);

        double alpha = 1.0 / omegahyperinvshape;
        double beta = alpha / omegahypermean;
        omegaarray->SetShape(alpha);
        omegaarray->SetScale(beta);
    }

    void MoveNucRates()    {

        ProfileMove(nucrelrate,0.1,1,3,&MultiGeneSingleOmegaModel::NucRatesLogProb,&MultiGeneSingleOmegaModel::TouchNucMatrix,this);
        ProfileMove(nucrelrate,0.03,3,3,&MultiGeneSingleOmegaModel::NucRatesLogProb,&MultiGeneSingleOmegaModel::TouchNucMatrix,this);
        ProfileMove(nucrelrate,0.01,3,3,&MultiGeneSingleOmegaModel::NucRatesLogProb,&MultiGeneSingleOmegaModel::TouchNucMatrix,this);

        ProfileMove(nucstat,0.1,1,3,&MultiGeneSingleOmegaModel::NucRatesLogProb,&MultiGeneSingleOmegaModel::TouchNucMatrix,this);
        ProfileMove(nucstat,0.01,1,3,&MultiGeneSingleOmegaModel::NucRatesLogProb,&MultiGeneSingleOmegaModel::TouchNucMatrix,this);
    }

    //-------------------
    // MPI send / receive
    //-------------------

    // global branch lengths
    
    void MasterSendGlobalBranchLengths() {
        MasterSendGlobal(*branchlength);
    }

    void SlaveReceiveGlobalBranchLengths()   {
        SlaveReceiveGlobal(*branchlength);
        for (int gene=0; gene<GetLocalNgene(); gene++)   {
            geneprocess[gene]->SetBranchLengths(*branchlength);
        }
    }

    // global nuc rates

    void MasterSendGlobalNucRates()   {
        MasterSendGlobal(nucrelrate,nucstat);
    }

    void SlaveReceiveGlobalNucRates()   {

        SlaveReceiveGlobal(nucrelrate,nucstat);
        for (int gene=0; gene<GetLocalNgene(); gene++)   {
            geneprocess[gene]->SetNucRates(nucrelrate,nucstat);
        }
    }

    // omega (and hyperparameters)

    void SlaveSendOmega()   {
        SlaveSendGeneArray(*omegaarray);
    }

    void MasterReceiveOmega()    {
        MasterReceiveGeneArray(*omegaarray);
    }

    void MasterSendOmega()  {
        MasterSendGeneArray(*omegaarray);
    }
    
    void SlaveReceiveOmega()    {
        SlaveReceiveGeneArray(*omegaarray);
        for (int gene=0; gene<GetLocalNgene(); gene++)    {
            geneprocess[gene]->SetOmega((*omegaarray)[gene]);
        }
    }

    // omega hyperparameters

    void MasterSendOmegaHyperParameters()   {
        MasterSendGlobal(omegahypermean,omegahyperinvshape);
    }

    void SlaveReceiveOmegaHyperParameters() {
        SlaveReceiveGlobal(omegahypermean,omegahyperinvshape);
        for (int gene=0; gene<GetLocalNgene(); gene++)    {
            geneprocess[gene]->SetOmegaHyperParameters(omegahypermean,omegahyperinvshape);
        }
    }

    // branch length suff stat

    void SlaveSendLengthSuffStat()  {
        lengthpathsuffstatarray->Clear();
        for (int gene=0; gene<GetLocalNgene(); gene++)   {
            geneprocess[gene]->CollectLengthSuffStat();
            lengthpathsuffstatarray->Add(*geneprocess[gene]->GetLengthPathSuffStatArray());
        }
        SlaveSendAdditive(*lengthpathsuffstatarray);
    }

    void MasterReceiveLengthSuffStat()  {

        lengthpathsuffstatarray->Clear();
        MasterReceiveAdditive(*lengthpathsuffstatarray);
    }

    // nuc rate suff stat

    void SlaveSendNucPathSuffStat()  {
        nucpathsuffstat.Clear();
        for (int gene=0; gene<GetLocalNgene(); gene++)   {
            geneprocess[gene]->CollectNucPathSuffStat();
            nucpathsuffstat += geneprocess[gene]->GetNucPathSuffStat();
        }
        SlaveSendAdditive(nucpathsuffstat);
    }

    void MasterReceiveNucPathSuffStat()  {
        nucpathsuffstat.Clear();
        MasterReceiveAdditive(nucpathsuffstat);
    }

    // log probs

    void SlaveSendLogProbs()   {

        GeneLogPrior = 0;
        lnL = 0;
        for (int gene=0; gene<GetLocalNgene(); gene++)   {
            GeneLogPrior += geneprocess[gene]->GetLogPrior();
            lnL += geneprocess[gene]->GetLogLikelihood();
        }
        SlaveSendAdditive(GeneLogPrior);
        SlaveSendAdditive(lnL);
    }

    void MasterReceiveLogProbs()    {

        GeneLogPrior = 0;
        MasterReceiveAdditive(GeneLogPrior);
        lnL = 0;
        MasterReceiveAdditive(lnL);
    }
};

