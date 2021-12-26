#pragma once

#include "CovMatrix.hpp"
#include "MultivariateNormalSuffStat.hpp"

class InverseWishart : public CovMatrix	{

	public:
    InverseWishart(size_t dim, int indf) :
        CovMatrix(dim), df(indf + dim) {}

    InverseWishart(const InverseWishart& from) :
        CovMatrix(from), df(from.df) {}

    virtual ~InverseWishart() {}

    double GetDiagLogDet(const std::vector<double>& kappa) const {
        double tot = 0;
        for (size_t i=0; i<size(); i++) {
            tot += log(kappa.at(i));
        }
        return tot;
    }

    double GetLogProb(const std::vector<double>& kappa) const   {
		if(isPositive()){
			double sum = 0;
            for (size_t i=0; i<size(); i++) {
				sum += GetInvMatrix()[i][i] * kappa.at(i);
			}
			double d = - ((GetLogDeterminant() * (size() + df + 1)) + sum) * 0.5;
			d += GetDiagLogDet(kappa) * df * 0.5;
			return d;
		}
		else{
            std::cerr << "singular cov matrix\n";
            exit(1);
			return -std::numeric_limits<double>::infinity();
		}
    }

    void Sample(const std::vector<double>& kappa)    {

        std::vector<std::vector<double>> iid(df, std::vector<double>(size(), 0));
		for (int i=0; i< df ; i++) {
            for (size_t j=0; j<size(); j++) {
                iid[i][j] = Random::sNormal()  / sqrt(kappa.at(j));
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
        std::vector<double> v(size(), 0);
        for (size_t i=0; i<size(); i++) {
			v[i] = Random::Gamma(0.5*(df-i),0.5);
		}
        std::vector<std::vector<double>> n(size(), std::vector<double>(size(), 0));
        std::vector<std::vector<double>> b(size(), std::vector<double>(size(), 0));
        std::vector<std::vector<double>> a(size(), std::vector<double>(size(), 0));

        for (size_t i=0; i<size(); i++) {
            for (size_t j=i+1; j<size(); j++) {
				n[i][j] = Random::sNormal();
			}
		}
        for (size_t i=0; i<size(); i++) {
			b[i][i] = v[i];
			for (size_t k=0; k<i; k++)	{
				b[i][i] += n[k][i] * n[k][i];
			}
		}
        for (size_t i=0; i<size(); i++) {
            for (size_t j=i+1; j<size(); j++) {
				b[i][j] = n[i][j] * sqrt(v[i]);
				for (size_t k=0; k<i; k++)	{
					b[i][j] += n[k][i] * n[k][j];
				}
				b[j][i] = b[i][j];
			}
		}

		A.CorruptDiag();
		A.Diagonalise();

		const std::vector<std::vector<double>>& p = A.GetEigenVect();
        const std::vector<double>& d = A.GetEigenVal();
		
        for (size_t i=0; i<size(); i++) {
            for (size_t j=0; j<size(); j++) {
				a[i][j] = p[i][j] / sqrt(d[j]);
			}
		}

        for (size_t i=0; i<size(); i++) {
            for (size_t j=0; j<size(); j++) {
				double tmp = 0;
				for (size_t k=0; k<size(); k++)	{
					tmp += b[i][k] * a[j][k];
				}
				n[i][j] = tmp;
			}
		}

        for (size_t i=0; i<size(); i++) {
            for (size_t j=0; j<size(); j++) {
				double tmp = 0;
				for (size_t k=0; k<size(); k++)	{
					tmp += a[i][k] * n[k][j];
				}
				setval(i, j, tmp);
			}
		}
		Invert();
	}

    void GibbsResample(const std::vector<double>& kappa, MultivariateNormalSuffStat& suffstat)    {
	    CovMatrix& scalestat = suffstat.covmat;
        int shapestat = suffstat.n;

        for (size_t i=0; i<size(); i++) {
			scalestat.add(i, i, kappa.at(i));
		}
		df += shapestat;
		scalestat.Diagonalise();
		SampleFromCovMatrix(scalestat);
        for (size_t i=0; i<size(); i++) {
			scalestat.add(i, i, -kappa.at(i));
		}
		df -= shapestat;
	}

    private:
    int df;
};

