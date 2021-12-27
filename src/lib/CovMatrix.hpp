#pragma once

#include "linalg.hpp"
#include "components/custom_tracer.hpp"

class CovMatrix : public custom_tracer {

	public:

	CovMatrix(int indim) : 
        value(indim, std::vector<double>(indim, 0)), 
        invvalue(indim, std::vector<double>(indim, 0)), 
        u(indim, std::vector<double>(indim, 0)), 
        invu(indim, std::vector<double>(indim, 0)), 
        v(indim,0),
        logv(indim,0),
        diagflag(false) {
            for (size_t i=0; i<size(); i++) {
                value[i][i] = 1.0;
            }
	}

    CovMatrix(const CovMatrix& from) : 
        value(from.value),
        invvalue(from.invvalue),
        u(from.u),
        invu(from.invu),
        v(from.v),
        logv(from.logv),
        diagflag(false) {
    }

    virtual ~CovMatrix() {}

    size_t size() const { return value.size(); }

    const std::vector<double>& operator[](size_t i) const {return  value.at(i);}

	double	operator()(int i, int j) const {
		return value.at(i).at(j);
	}

    void to_stream_header(std::string name, std::ostream& os) const override {
        for (size_t i=0; i<size(); i++) {
            for (size_t j=i+1; j<size(); j++)   {
                os << name << "[" << i << "][" << j << "]" << '\t';
            }
        }
        for (size_t i=0; i<size(); i++) {
            os << name << "[" << i << "][" << i << "]";
            if (i<size()-1) {
                os << '\t';
            }
        }
    }

    void to_stream(std::ostream& os) const override {
        for (size_t i=0; i<size(); i++) {
            for (size_t j=i+1; j<size(); j++)   {
                os << value[i][j] << '\t';
            }
        }
        for (size_t i=0; i<size(); i++) {
            os << value[i][i];
            if (i<size()-1) {
                os << '\t';
            }
        }
    }

    void from_stream(std::istream& is) override {
        for (size_t i=0; i<size(); i++) {
            for (size_t j=i+1; j<size(); j++)   {
                is >> value[i][j];
                value[j][i] = value[i][j];
            }
        }
        for (size_t i=0; i<size(); i++) {
            is >>  value[i][i];
        }
        CorruptDiag();
    }

    void setval(int i, int j, double val)   {
        value[i][j] = val;
        diagflag = false;
    }

    void add(int i, int j, double val)  {
        value[i][j] += val;
        diagflag = false;
    }

	CovMatrix& operator*=(double scal) {
        for (size_t i=0; i<size(); i++) {
            for (size_t j=0; j<size(); j++) {
                value[i][j] *= scal;
                std::cerr << value[i][j] << '\t';
            }
            std::cerr << '\n';
        }
		diagflag = false;
		return *this;
	}

	virtual void MultivariateNormalSample(std::vector<double>& vec, bool inverse=false) const {
        std::vector<double> principalcomp(size(), 0);
        for (size_t i=0; i<size(); i++) {
            if (inverse)    {
                principalcomp[i] =  Random::sNormal() * sqrt(GetEigenVal()[i]);
            }
            else    {
                principalcomp[i] =  Random::sNormal() / sqrt(GetEigenVal()[i]);
            }
		}
        for (size_t i=0; i<size(); i++) {
			vec[i]=0;
        }
        for (size_t i=0; i<size(); i++) {
            for (size_t j=0; j<size(); j++) {
				vec[j] +=  principalcomp[i] * GetEigenVect()[j][i];
			}
		}
	}

	double logMultivariateNormalDensity(const std::vector<double>& dval) const {

		double tXSX = 0;
        for (size_t i=0; i<size(); i++) {
			tXSX += GetInvMatrix()[i][i] * dval[i] * dval[i];
			for (size_t j=0; j<i ; j++) {
				tXSX += 2 * GetInvMatrix()[i][j] * dval[j] * dval[i];
			}
		}
		return -0.5 * (GetLogDeterminant() + tXSX);
	}

	const std::vector<double>& GetEigenVal() const {
		if (! diagflag) Diagonalise();
		return v;
	}

	const std::vector<double>& GetLogEigenVal() const {
		if (! diagflag) Diagonalise();
		return logv;
	}

	const std::vector<std::vector<double>>& GetEigenVect() const {
		if (! diagflag)	{
			Diagonalise();
		}
		return u;
	}

	double GetLogDeterminant() const {
		double ret = 0;
        for (size_t i=0; i<size(); i++) {
			ret += GetLogEigenVal()[i];
		}
		return ret;
	}

	const std::vector<std::vector<double>>& GetInvEigenVect() const {
		if (! diagflag) Diagonalise();
		return invu;
	}

	void SetToZero()	{
        for (size_t i=0; i<size(); i++) {
            for (size_t j=0; j<size(); j++) {
				value[i][j] = 0;
			}
		}
        diagflag = false;
	}

	void SetToIdentity()	{
        for (size_t i=0; i<size(); i++) {
            for (size_t j=0; j<size(); j++) {
				value[i][j] = 0;
			}
		}
        for (size_t i=0; i<size(); i++) {
			value[i][i] = 1;
		}
        diagflag = false;
	}

    /*
	void Project(int index, double** m)	{

		int k = 0;
        for (size_t i=0; i<size(); i++) {
			if (i != index)	{
				size_t l = 0;
                for (size_t j=0; j<size(); j++) {
					if (j != index)	{
						m[k][l] = value[i][j] - value[i][index] * value[j][index] / value[index][index];
						l++;
					}
				}
				k++;
			}
		}
	}
    */

	const std::vector<std::vector<double>>& GetInvMatrix() const {
		if (! diagflag) Diagonalise();
		return invvalue;
	}

	bool isPositive() const {
		if (! diagflag) Diagonalise();
		bool r = true;
        for (size_t i=0; i<size(); i++) {
			if(GetEigenVal()[i] <= 1e-6){
				r = false;
			}
		}
		return r;
	}

	void CorruptDiag() const {
		diagflag = false;
	}

	double GetMax() const {
		double max = 0;
        for (size_t i=0; i<size(); i++) {
            for (size_t j=0; j<size(); j++) {
				double tmp = fabs(value.at(i).at(j));
				if (max < tmp)	{
					max = tmp;
				}
			}
		}
		return max;
	}

	//Set the matrix to it s inverse //loook si diagflag
	int Invert() {
        std::vector<std::vector<double>> a(size(), std::vector<double>(size(), 0));

		// copy value into a :
        for (size_t i=0; i<size(); i++) {
            for (size_t j=0; j<size(); j++) {
				a[i][j] = value[i][j];
			}
		}

		double logdet = LinAlg::Gauss(a,size(), value);

		diagflag = false;
		if (std::isinf(logdet))	{
			std::cerr << "error in cov matrix: non invertible\n";
			return 1;
			exit(1);
		}
		return 0;
	}

	double CheckInverse() const {
		double max = 0;
        for (size_t i=0; i<size(); i++) {
            for (size_t j=0; j<size(); j++) {
				double tot = 0;
                for (size_t k=0; k<size(); k++) {
					tot += value.at(i).at(k) * GetInvMatrix()[k][j];
				}
				if (i == j)	{
					tot --;
				}
				if (max < fabs(tot))	{
					max = fabs(tot);
				}
			}
		}
		return max;
	}

	int Diagonalise() const {

		int nmax = 1000;
		double epsilon = 1e-10;

		int n = LinAlg::DiagonalizeSymmetricMatrix(value,size(),nmax,epsilon,v,u);
		bool failed = (n == nmax);
		if (failed)	{
			std::cerr << "diag failed\n";
			std::cerr << n << '\n';
            for (size_t i=0; i<size(); i++) {
                for (size_t j=0; j<size(); j++) {
					std::cerr << value.at(i).at(j) << '\t';
				}
				std::cerr << '\n';
			}
			exit(1);
		}

		// normalise u
        for (size_t i=0; i<size(); i++) {
			double total = 0;
            for (size_t j=0; j<size(); j++) {
				total += u[j][i] * u[j][i];
			}
		}
		// u-1 = tu
        for (size_t i=0; i<size(); i++) {
            for (size_t j=0; j<size(); j++) {
				invu[j][i] = u[i][j];
			}
		}

        for (size_t i=0; i<size(); i++) {
			logv[i] = log(v[i]);
		}

		LinAlg::Gauss(value,size(),invvalue);

		diagflag = true;
		double tmp = CheckDiag();
        if (tmp > 1e-6) {
            std::cerr << "diag error in cov matrix\n";
            std::cerr << tmp << '\n';
            exit(1);
        }
		return failed;
	}

	double CheckDiag() const {
        std::vector<std::vector<double>> a(size(), std::vector<double>(size(), 0));
        std::vector<std::vector<double>> b(size(), std::vector<double>(size(), 0));

        for (size_t i=0; i<size(); i++) {
            for (size_t j=0; j<size(); j++) {
				double tot = 0;
                for (size_t k=0; k<size(); k++) {
					tot += invu[i][k] * value.at(k).at(j);
				}
				a[i][j] = tot;
			}
		}

		double max = 0;

        for (size_t i=0; i<size(); i++) {
            for (size_t j=0; j<size(); j++) {
				double tot = 0;
                for (size_t k=0; k<size(); k++) {
					tot += a[i][k] * u[k][j];
				}
				b[i][j] = tot;
				if (i != j)	{
					if (max < fabs(tot))	{
						max = fabs(tot);
					}
				}
			}
		}
		return max;
	}

	void SetToScatterMatrix(const std::vector<std::vector<double>>& invals)    {
        int df = invals.size();
        for (size_t i=0; i<size(); i++) {
            for (size_t j=0; j<size(); j++) {
				value[i][j] = 0;
				for (int l=0; l<df; l++) {
					value[i][j] += invals[l][i] * invals[l][j];
				}
			}
		}
		diagflag = false;
	}

	private:

	std::vector<std::vector<double>> value;

    // inverse
	mutable std::vector<std::vector<double>> invvalue;
    // eigenvect
	mutable std::vector<std::vector<double>> u;
    // inv eigenvect
	mutable std::vector<std::vector<double>> invu;
    // eigenval
    mutable std::vector<double> v;
    // log eigenval
    mutable std::vector<double> logv;
	mutable bool diagflag;
};

