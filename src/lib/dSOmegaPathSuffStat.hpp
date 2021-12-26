#pragma once

#include "PathSuffStat.hpp"
#include "RelativePathSuffStat.hpp"
#include "CodonSubMatrix.hpp"
#include "PoissonSuffStat.hpp"

class dSOmegaPathSuffStat {

    public:

    dSOmegaPathSuffStat() {
        Clear();
    }
    ~dSOmegaPathSuffStat() {}

    void Clear()    {
        nsyn = nnonsyn = 0;
        bsyn = bnonsyn = 0;
    }

    void AddSuffStat(const OmegaCodonSubMatrix &codonsubmatrix, const PathSuffStat &pathsuffstat, double branchlength, double omega) {
        int ncodon = codonsubmatrix.GetNstate();
        const CodonStateSpace *statespace = codonsubmatrix.GetCodonStateSpace();

        const std::map<std::pair<int, int>, int> &paircount = pathsuffstat.GetPairCountMap();
        const std::map<int, double> &waitingtime = pathsuffstat.GetWaitingTimeMap();

        double tmpbsyn = 0;
        double tmpbnonsyn = 0;
        for (std::map<int, double>::const_iterator i = waitingtime.begin(); i != waitingtime.end();
             i++) {
            double totsynrate = 0;
            double totnonsynrate = 0;
            int a = i->first;
            for (int b = 0; b < ncodon; b++) {
                if (b != a) {
                    if (codonsubmatrix(a, b) != 0) {
                        if (!statespace->Synonymous(a, b)) {
                            totnonsynrate += codonsubmatrix(a, b);
                        }
                        else    {
                            totsynrate += codonsubmatrix(a, b);
                        }
                    }
                }
            }
            tmpbsyn += i->second * totsynrate;
            tmpbnonsyn += i->second * totnonsynrate;
        }
        tmpbsyn /= branchlength;
        tmpbnonsyn /= branchlength*omega;
        bsyn += tmpbsyn;
        bnonsyn += tmpbnonsyn;

        for (std::map<std::pair<int, int>, int>::const_iterator i = paircount.begin(); i != paircount.end(); i++) {
            if (!statespace->Synonymous(i->first.first, i->first.second)) {
                nnonsyn += i->second;
            }
            else    {
                nsyn += i->second;
            }
        }
    }

    void AddSuffStat(const OmegaCodonSubMatrix &codonsubmatrix, const RelativePathSuffStat &pathsuffstat, double omega) {
        int ncodon = codonsubmatrix.GetNstate();
        const CodonStateSpace *statespace = codonsubmatrix.GetCodonStateSpace();

        const std::map<std::pair<int, int>, int> &paircount = pathsuffstat.GetPairCountMap();
        const std::map<int, double> &waitingtime = pathsuffstat.GetWaitingTimeMap();

        double tmpbsyn = 0;
        double tmpbnonsyn = 0;
        for (std::map<int, double>::const_iterator i = waitingtime.begin(); i != waitingtime.end();
             i++) {
            double totsynrate = 0;
            double totnonsynrate = 0;
            int a = i->first;
            for (int b = 0; b < ncodon; b++) {
                if (b != a) {
                    if (codonsubmatrix(a, b) != 0) {
                        if (!statespace->Synonymous(a, b)) {
                            totnonsynrate += codonsubmatrix(a, b);
                        }
                        else    {
                            totsynrate += codonsubmatrix(a, b);
                        }
                    }
                }
            }
            tmpbsyn += i->second * totsynrate;
            tmpbnonsyn += i->second * totnonsynrate;
        }
        tmpbnonsyn /= omega;
        bsyn += tmpbsyn;
        bnonsyn += tmpbnonsyn;

        for (std::map<std::pair<int, int>, int>::const_iterator i = paircount.begin(); i != paircount.end(); i++) {
            if (!statespace->Synonymous(i->first.first, i->first.second)) {
                nnonsyn += i->second;
            }
            else    {
                nsyn += i->second;
            }
        }
    }

    double GetLogProb(double l, double omega) const { 
        return (nsyn + nnonsyn)*log(l) + nnonsyn*log(omega) - l*(bsyn + bnonsyn*omega);
    }

    double GetLogProbdSIntegrated(double l, double omega, double dt, double nu) const   {
        //double alpha = dt / nu;
        double alpha = 1.0 / nu;
        double alphapost = alpha + nsyn + nnonsyn;
        double betapost = alpha + l*(bsyn + bnonsyn*omega);
        return alpha*log(alpha) - Random::logGamma(alpha) - alphapost*log(betapost) + Random::logGamma(alphapost) + (nsyn + nnonsyn)*log(l) + nnonsyn*log(omega);
    }

    double GetLogProbOmIntegrated(double l, double omega, double dt, double nu) const   {
        // double alpha = dt / nu;
        double alpha = 1.0 / nu;
        double alphapost = alpha + nnonsyn;
        double betapost = alpha + l*bnonsyn*omega;
        return alpha*log(alpha) - Random::logGamma(alpha) - alphapost*log(betapost) + Random::logGamma(alphapost) + (nsyn + nnonsyn)*log(l) + nnonsyn*log(omega) - l*bsyn;
    }

    void Add(const dSOmegaPathSuffStat &from) {
        nsyn += from.nsyn;
        nnonsyn += from.nnonsyn;
        bsyn += from.bsyn;
        bnonsyn += from.bnonsyn;
    }

    void Add(double syncount, double synbeta, double nonsyncount, double nonsynbeta)    {
        nsyn += syncount;
        nnonsyn += nonsyncount;
        bsyn += synbeta;
        bnonsyn += nonsynbeta;
    }

    void TodSSuffStat(PoissonSuffStat& suffstat, double omega) const  {
        suffstat.AddSuffStat(nsyn + nnonsyn, bsyn + omega*bnonsyn);
    }

    void ToOmSuffStat(PoissonSuffStat& suffstat, double l) const  {
        suffstat.AddSuffStat(nnonsyn, l*bnonsyn);
    }

    void AddWNdSSuffStat(PoissonSuffStat& suffstat, double l, double omega) const   {
        suffstat.AddSuffStat(nsyn + nnonsyn, l*(bsyn + bnonsyn*omega));
    }

    void AddWNOmSuffStat(PoissonSuffStat& suffstat, double l, double omega) const   {
        suffstat.AddSuffStat(nnonsyn, l*omega*bnonsyn);
    }

    double GetCount() const {
        return nsyn + nnonsyn;
    }

    double GetSynCount() const {
        return nsyn;
    }

    double GetNonSynCount() const  {
        return nnonsyn;
    }

    double GetSynBeta() const  {
        return bsyn;
    }

    double GetNonSynBeta() const    {
        return bnonsyn;
    }

    double GetBeta(double omega) const  {
        return bsyn + omega*bnonsyn;
    }

    dSOmegaPathSuffStat &operator+=(const dSOmegaPathSuffStat &from) {
        Add(from);
        return *this;
    }

    double GetdNdS() const    {
        if ((!bsyn) || (!bnonsyn) || (!nsyn))    {
            return 0;
        }
        return (nnonsyn / bnonsyn) / (nsyn / bsyn);
    }

    double GetdS() const    {
        if (! bsyn) {
            return 0;
        }
        return nsyn / bsyn;
    }

    double GetdN() const    {
        if (! bnonsyn)  {
            return 0;
        }
        return nnonsyn / bnonsyn;
    }

    void Normalize(double factor)   {
        nsyn *= factor;
        nnonsyn *= factor;
        bsyn *= factor;
        bnonsyn *= factor;
    }
        
    private:

    double nsyn;
    double nnonsyn;
    double bsyn;
    double bnonsyn;
};

