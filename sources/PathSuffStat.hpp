
#ifndef PATHSUFFSTAT_H
#define PATHSUFFSTAT_H

#include "SuffStat.hpp"
#include "PoissonSuffStat.hpp"
#include "CodonSubMatrix.hpp"
#include "SubMatrix.hpp"
#include "CodonSubMatrix.hpp"
#include "CodonSubMatrixArray.hpp"
#include "Array.hpp"
#include <map>

class PathSuffStat : public SuffStat	{

	public:

	PathSuffStat() {}
	~PathSuffStat() {}

	void Clear()	{
		rootcount.clear();
		paircount.clear();
		waitingtime.clear();
	}

	void IncrementRootCount(int state)	{
		rootcount[state]++;
	}

	void IncrementPairCount(int state1, int state2)	{
		paircount[pair<int,int>(state1,state2)]++;
	}

	void AddRootCount(int state, int in)	{
		rootcount[state] += in;
	}

	void AddPairCount(int state1, int state2, int in)	{
		paircount[pair<int,int>(state1,state2)] += in;
	}
	
	void AddWaitingTime(int state, double in)	{
		waitingtime[state] += in;
	}

	int GetRootCount(int state) const {
		return rootcount[state];
	}

	int GetPairCount(int state1, int state2) const  {
		return paircount[pair<int,int>(state1,state2)];
	}

	double GetWaitingTime(int state) const	{
		return waitingtime[state];
	}
	
	double GetLogProb(const SubMatrix& mat) const {
		double total = 0;
		const double* stat = mat.GetStationary();
		for (std::map<int,int>::iterator i = rootcount.begin(); i!= rootcount.end(); i++)	{
			total += i->second * log(stat[i->first]);
		}
		for (std::map<int,double>::iterator i = waitingtime.begin(); i!= waitingtime.end(); i++)	{
			total += i->second * mat(i->first,i->first);
		}
		for (std::map<pair<int,int>, int>::iterator i = paircount.begin(); i!= paircount.end(); i++)	{
			total += i->second * log(mat(i->first.first, i->first.second));
		}
		return total;
	}

	void AddOmegaSuffStat(PoissonSuffStat& omegasuffstat, const MGOmegaCodonSubMatrix& matrix) const {

		int ncodon = matrix.GetNstate();
		const CodonStateSpace* statespace = matrix.GetCodonStateSpace();

		double beta = 0;
		for (std::map<int,double>::iterator i = waitingtime.begin(); i!= waitingtime.end(); i++)	{
			double totnonsynrate = 0;
			int a = i->first;
			for (int b=0; b<ncodon; b++)	{
				if (b != a)	{
					if (matrix(a,b) != 0)	{
						if (!statespace->Synonymous(a,b))	{
							totnonsynrate += matrix(a,b);
						}
					}
				}
			}
			beta += i->second * totnonsynrate;
		}
		beta /= matrix.GetOmega();

		int count = 0;
		for (std::map<pair<int,int>, int>::iterator i = paircount.begin(); i!= paircount.end(); i++)	{
			if (! statespace->Synonymous(i->first.first,i->first.second))	{
				count += i->second;
			}
		}
		omegasuffstat.AddSuffStat(count,beta);
	}

	private:

	mutable std::map<int,int> rootcount;
	mutable std::map<pair<int,int>,int> paircount;
	mutable std::map<int,double> waitingtime;
};

class PathSuffStatArray : public SimpleArray<PathSuffStat>	{

	public:

	PathSuffStatArray(int insize) : SimpleArray<PathSuffStat>(insize) {}
	~PathSuffStatArray() {}

	void Clear()	{
		for (int i=0; i<GetSize(); i++)	{
			(*this)[i].Clear();
		}
	}

	double GetLogProb(const Array<SubMatrix>* matrixarray) const	{

		double total = 0;
		for (int i=0; i<GetSize(); i++)	{
			total += GetVal(i).GetLogProb(matrixarray->GetVal(i));
		}
		return total;
	}

    /*
    void AddOmegaSuffStat(PoissonSuffStatArray* omegasuffstatarray, const MGOmegaHeterogeneousCodonSubMatrixArray* matrixarray) const {
		for (int i=0; i<GetSize(); i++)	{
                GetVal(i).AddOmegaSuffStat(omegasuffstatarray->GetOmegaSuffStat(i),matrixarray->GetMGOmegaCodonSubMatrix(i));
                // GetVal(i).AddOmegaSuffStat((*omegasuffstatarray)[i],matrixarray->GetMGOmegaCodonSubMatrix(i));
        }
    }
    */
};

#endif
