
#ifndef IIDGAMMA_H
#define IIDGAMMA_H

#include "Array.hpp"
#include "BranchArray.hpp"
#include "Random.hpp"
#include "PoissonSuffStat.hpp"

class GammaSuffStat : public SuffStat	{

	public:
	GammaSuffStat() {}
	~GammaSuffStat() {}

	void Clear()	{
		sum = 0;
		sumlog = 0;
		n = 0;
	}

	void AddSuffStat(double x, double logx, int c = 1)	{
		sum += x;
		sumlog += logx;
		n += c;
	}

	double GetLogProb(double shape, double scale) const {
		return n*(shape*log(scale) - Random::logGamma(shape)) + (shape-1)*sumlog - scale*sum;
	}
	
    double GetSum() const {return sum;}
    double GetSumLog() const {return sumlog;}
    int GetN() const {return n;}

	private:

	double sum;
	double sumlog;
	int n;
};

class GammaSuffStatBranchArray : public SimpleBranchArray<GammaSuffStat>    {

    public:
	GammaSuffStatBranchArray(const Tree& intree) : SimpleBranchArray<GammaSuffStat>(intree) {}
    ~GammaSuffStatBranchArray() {}

    void Clear()    {
        for (int i=0; i<GetNbranch(); i++)  {
            (*this)[i].Clear();
        }
    }

    double GetLogProb(const ConstBranchArray<double>& blmean, double invshape) const {

        double total = 0;
        for (int i=0; i<GetNbranch(); i++)  {
            double shape = 1.0 / invshape;
            double scale = 1.0 / blmean.GetVal(i);
            total += GetVal(i).GetLogProb(shape,scale);
        }
        return total;
    }
};

class IIDGamma: public SimpleArray<double>	{

	public: 

	IIDGamma(int insize, double inshape, double inscale) : SimpleArray<double>(insize), shape(inshape), scale(inscale)	{
		Sample();
	}

	~IIDGamma() {}

	double GetShape() const {return shape;}
	double GetScale() const {return scale;}

	void SetShape(double inshape)	{
		shape = inshape;
	}

	void SetScale(double inscale)	{
		scale = inscale;
	}

	void Sample()	{
		for (int i=0; i<GetSize(); i++)	{
			(*this)[i] = Random::GammaSample(shape,scale);
		}
	}

	void GibbsResample(const ConstArray<PoissonSuffStat>& suffstatarray)	{
		for (int i=0; i<GetSize(); i++)	{
			const PoissonSuffStat& suffstat = suffstatarray.GetVal(i);
			(*this)[i] = Random::GammaSample(shape + suffstat.GetCount(), scale + suffstat.GetBeta());
		}
	}

	double GetLogProb()	const {
		double total = 0;
		for (int i=0; i<GetSize(); i++)	{
			total += GetLogProb(i);
		}
		return total;
	}

	double GetLogProb(int index) const {
        return Random::logGammaDensity(GetVal(index),shape,scale);
	}

	void AddSuffStat(GammaSuffStat& suffstat) const {
		for (int i=0; i<GetSize(); i++)	{
			suffstat.AddSuffStat(GetVal(i),log(GetVal(i)));
		}
	}

	void AddSuffStat(GammaSuffStat& suffstat, const ConstArray<double>& poswarray) const {
		for (int i=0; i<GetSize(); i++)	{
            if (poswarray.GetVal(i)) {
                suffstat.AddSuffStat(GetVal(i),log(GetVal(i)));
            }
		}
	}

    void PriorResample(const ConstArray<double>& poswarray) {
		for (int i=0; i<GetSize(); i++)	{
            if (poswarray.GetVal(i)) {
                (*this)[i] = Random::GammaSample(shape,scale);
            }
		}
    }

    double GetMean() const {
        double m1 = 0;
        for (int i=0; i<GetSize(); i++) {
            m1 += GetVal(i);
        }
        m1 /= GetSize();
        return m1;
    }

    double GetVar() const {
        double m1 = 0;
        double m2 = 0;
        for (int i=0; i<GetSize(); i++) {
            m1 += GetVal(i);
            m2 += GetVal(i) * GetVal(i);
        }
        m1 /= GetSize();
        m2 /= GetSize();
        m2 -= m1*m1;
        return m2;
    }

	protected:
	double shape;
	double scale;
};

class BranchIIDGamma: public SimpleBranchArray<double>	{

	public: 

	BranchIIDGamma(const Tree& intree, double inshape, double inscale) : SimpleBranchArray<double>(intree), shape(inshape), scale(inscale)	{
		Sample();
	}

	~BranchIIDGamma() {}

	double GetShape() const {return shape;}
	double GetScale() const {return scale;}

	void SetShape(double inshape)	{
		shape = inshape;
	}

	void SetScale(double inscale)	{
		scale = inscale;
	}

	void SetAllBranches(double inval)	{
		for (int i=0; i<GetNbranch(); i++)	{
			(*this)[i] = inval;
		}
	}

	void Sample()	{
		for (int i=0; i<GetNbranch(); i++)	{
			(*this)[i] = Random::GammaSample(shape,scale);
		}
	}

	void GibbsResample(const PoissonSuffStatBranchArray& suffstatarray)	{
		for (int i=0; i<GetNbranch(); i++)	{
			const PoissonSuffStat& suffstat = suffstatarray.GetVal(i);
			(*this)[i] = Random::GammaSample(shape + suffstat.GetCount(), scale + suffstat.GetBeta());
		}
	}

	double GetLogProb()	{
		double total = 0;
		for (int i=0; i<GetNbranch(); i++)	{
			total += GetLogProb(i);
		}
		return total;
	}

	double GetLogProb(int index)	{
        return Random::logGammaDensity(GetVal(index),shape,scale);
	}

	void AddSuffStat(GammaSuffStat& suffstat)	{
		for (int i=0; i<GetNbranch(); i++)	{
			suffstat.AddSuffStat((*this)[i],log((*this)[i]));
		}
	}

    /*
    void AddSuffStat(GammaSuffStatBranchArray& suffstatarray)   {
		for (int i=0; i<GetNbranch(); i++)	{
			suffstat[i].AddSuffStat((*this)[i],log((*this)[i]));
		}
    }
    */

	protected:
	double shape;
	double scale;
};

class GammaWhiteNoise: public SimpleBranchArray<double>	{

	public: 

	GammaWhiteNoise(const Tree& intree, const ConstBranchArray<double>& inblmean, double inshape) : SimpleBranchArray<double>(intree), blmean(inblmean), shape(inshape)	{
		Sample();
	}

	~GammaWhiteNoise() {}

	double GetShape() const {return shape;}

	void SetShape(double inshape)	{
		shape = inshape;
	}

	void Sample()	{
		for (int i=0; i<GetNbranch(); i++)	{
            double scale = shape / blmean.GetVal(i);
			(*this)[i] = Random::GammaSample(shape,scale);
		}
	}

	void GibbsResample(const PoissonSuffStatBranchArray& suffstatarray)	{
		for (int i=0; i<GetNbranch(); i++)	{
			const PoissonSuffStat& suffstat = suffstatarray.GetVal(i);
            double scale = shape / blmean.GetVal(i);
			(*this)[i] = Random::GammaSample(shape + suffstat.GetCount(), scale + suffstat.GetBeta());
		}
	}

	double GetLogProb()	{
		double total = 0;
		for (int i=0; i<GetNbranch(); i++)	{
			total += GetLogProb(i);
		}
		return total;
	}

	double GetLogProb(int index) const {
        double scale = shape / blmean.GetVal(index);
        return Random::logGammaDensity(GetVal(index),shape,scale);
	}

	void AddSuffStat(GammaSuffStat& suffstat)	{
		for (int i=0; i<GetNbranch(); i++)	{
			suffstat.AddSuffStat((*this)[i],log((*this)[i]));
		}
	}

    void AddSuffStat(GammaSuffStatBranchArray& suffstatarray)   {
		for (int i=0; i<GetNbranch(); i++)	{
			suffstatarray[i].AddSuffStat((*this)[i],log((*this)[i]));
		}
    }

	protected:
    const ConstBranchArray<double>& blmean;
	double shape;
};

class GammaWhiteNoiseArray : public Array<GammaWhiteNoise>    {

    public:

    GammaWhiteNoiseArray(int inNgene, const Tree& intree, const ConstBranchArray<double>& inblmean, double inshape) : Ngene(inNgene), tree(intree), blmean(inblmean), shape(inshape), blarray(Ngene, (GammaWhiteNoise*) 0)    {
        
        for (int gene=0; gene<Ngene; gene++)    {
            blarray[gene] = new GammaWhiteNoise(tree,blmean,shape);
        }
    }

    ~GammaWhiteNoiseArray()  {
        for (int gene=0; gene<Ngene; gene++)    {
            delete blarray[gene];
        }
    }

    void SetShape(double inshape)   {
        shape = inshape;
        for (int gene=0; gene<Ngene; gene++)    {
            blarray[gene]->SetShape(shape);
        }
    }

    int GetSize() const {
        return Ngene;
    }

    const GammaWhiteNoise& GetVal(int gene) const {
        return *blarray[gene];
    }

    GammaWhiteNoise& operator[](int gene)  {
        return *blarray[gene];
    }

    int GetNgene() const {
        return Ngene;
    }

    int GetNbranch() const {
        return blarray[0]->GetNbranch();
    }

    double GetMeanLength() const {
        double tot = 0;
        for (int j=0; j<GetNbranch(); j++)   {
            double mean = 0;
            for (int g=0; g<GetNgene(); g++) {
                double tmp = blarray[g]->GetVal(j);
                mean += tmp;
            }
            mean /= GetNgene();
            tot += mean;
        }
        return tot;
    }

    double GetVarLength() const {
        double tot = 0;
        for (int j=0; j<GetNbranch(); j++)   {
            double mean = 0;
            double var = 0;
            for (int g=0; g<GetNgene(); g++) {
                double tmp = blarray[g]->GetVal(j);
                mean += tmp;
                var += tmp*tmp;
            }
            mean /= GetNgene();
            var /= GetNgene();
            var -= mean*mean;
            tot += var;
        }
        tot /= GetNbranch();
        return tot;
    }

    double GetLogProb() const {
        double total = 0;
        for (int gene=0; gene<GetNgene(); gene++)  {
            total += blarray[gene]->GetLogProb();
        }
        return total;
    }

    void AddSuffStat(GammaSuffStatBranchArray& suffstatarray) const    {
        for (int gene=0; gene<GetNgene(); gene++)  {
            blarray[gene]->AddSuffStat(suffstatarray);
        }
    }

    private:

    int Ngene;
    const Tree& tree;
    const ConstBranchArray<double>& blmean;
	double shape;
    vector<GammaWhiteNoise*> blarray;
};

#endif
