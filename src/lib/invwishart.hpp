#pragma once

#include "CovMatrix.hpp"
#include "MultivariateNormalSuffStat.hpp"

class InverseWishart : public CovMatrix	{

	public:
    InverseWishart(const std::vector<double>& inkappa, int indf) :
        CovMatrix(inkappa.size()), kappa(inkappa), df(indf + inkappa.size()) {
            Sample();
    }

    InverseWishart(const InverseWishart& from) :
        CovMatrix(from), kappa(from.kappa), df(from.df) {}

    virtual ~InverseWishart() {}

    /*
    void SetDF(int indf)    {
        df = indf;
    }
    */

    void SetKappa(const std::vector<double>& inkappa)    {
        assert(kappa.size() == inkappa.size());
        for (size_t i=0; i<kappa.size(); i++)   {
            kappa[i] = inkappa[i];
        }
    }

    void Corrupt()  {
        CorruptDiag();
    }


    int GetDim() const  {
        return int(kappa.size());
    }

    double GetDiagLogDet() const    {
        double total = 0;
        for (int i=0; i<GetDim(); i++)  {
            total += log(kappa[i]);
        }
        return total;
    }

    double GetLogProb() const   {
		if(isPositive()){
			double sum = 0;
			for(int i=0; i<GetDim(); i++){
				sum += GetInvMatrix()[i][i] * kappa[i];
			}
			double d = - ((GetLogDeterminant() * (GetDim() + df + 1)) + sum) * 0.5;
			d += GetDiagLogDet() * df * 0.5;
			return d;
		}
		else{
            std::cerr << "singular cov matrix\n";
            exit(1);
			return -std::numeric_limits<double>::infinity();
		}
    }

    void Sample()    {

        std::vector<std::vector<double>> iid(df, std::vector<double>(GetDim(), 0));
		for (int i=0; i< df ; i++) {
            for (int j=0; j<GetDim(); j++) {
                iid[i][j] = Random::sNormal()  / sqrt(kappa[j]);
            }
		}

		SetToScatterMatrix(iid);
		int ret = Invert();
        if (ret)   {
            std::cerr << "matrix inversion error in inverse wishart\n";
            exit(1);
        }
    }

    void SampleFromCovMatrix(const CovMatrix& A)  {

		// algorithm of Odell and Feiveson, 1966
        std::vector<double> v(GetDim(), 0);
		for (int i=0; i<GetDim(); i++)	{
			v[i] = Random::Gamma(0.5*(df-i),0.5);
		}
        std::vector<std::vector<double>> n(GetDim(), std::vector<double>(GetDim(), 0));
        std::vector<std::vector<double>> b(GetDim(), std::vector<double>(GetDim(), 0));
        std::vector<std::vector<double>> a(GetDim(), std::vector<double>(GetDim(), 0));

		for (int i=0; i<GetDim(); i++)	{
			for (int j=i+1; j<GetDim(); j++)	{
				n[i][j] = Random::sNormal();
			}
		}
		for (int i=0; i<GetDim(); i++)	{
			b[i][i] = v[i];
			for (int k=0; k<i; k++)	{
				b[i][i] += n[k][i] * n[k][i];
			}
		}
		for (int i=0; i<GetDim(); i++)	{
			for (int j=i+1; j<GetDim(); j++)	{
				b[i][j] = n[i][j] * sqrt(v[i]);
				for (int k=0; k<i; k++)	{
					b[i][j] += n[k][i] * n[k][j];
				}
				b[j][i] = b[i][j];
			}
		}

		A.CorruptDiag();
		A.Diagonalise();

		const std::vector<std::vector<double>>& p = A.GetEigenVect();
        // ??
		// const std::vector<std::vector<double>>& invp = A.GetInvEigenVect();
        const std::vector<double>& d = A.GetEigenVal();
		
		for (int i=0; i<GetDim(); i++)	{
			for(int j=0; j<GetDim(); j++)	{
				a[i][j] = p[i][j] / sqrt(d[j]);
			}
		}

		for (int i=0; i<GetDim(); i++)	{
			for(int j=0; j<GetDim(); j++)	{
				double tmp = 0;
				for (int k=0; k<GetDim(); k++)	{
					tmp += b[i][k] * a[j][k];
				}
				n[i][j] = tmp;
			}
		}

		for (int i=0; i<GetDim(); i++)	{
			for(int j=0; j<GetDim(); j++)	{
				double tmp = 0;
				for (int k=0; k<GetDim(); k++)	{
					tmp += a[i][k] * n[k][j];
				}
				setval(i, j, tmp);
			}
		}
		Invert();
	}

    void GibbsResample(MultivariateNormalSuffStat& suffstat)    {
	    CovMatrix& scalestat = suffstat.covmat;
        int shapestat = suffstat.n;

		for(int i=0; i<GetDim(); i++){
			scalestat.add(i, i, kappa[i]);
		}
		df += shapestat;
		scalestat.Diagonalise();
		SampleFromCovMatrix(scalestat);
		for(int i=0; i<GetDim(); i++){
			scalestat.add(i, i, -kappa[i]);
		}
		df -= shapestat;
	}

    private:
    std::vector<double> kappa;
    int df;
};

