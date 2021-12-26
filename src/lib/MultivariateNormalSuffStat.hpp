#pragma once

#include "CovMatrix.hpp"
#include "MultivariateBrownianTreeProcess.hpp"

class MultivariateNormalSuffStat {

    public:

    MultivariateNormalSuffStat(size_t indim) : covmat(indim), n(0) {}

    void Clear()    {
        for (size_t i=0; i<covmat.size(); i++) {
            for (size_t j=0; j<covmat.size(); j++) {
                covmat.setval(i,j,0);
            }
        }
        n = 0;
    }

    void AddSuffStat(const MultivariateBrownianTreeProcess& process)    {
        process.GetSampleCovarianceMatrix(covmat, n);
    }

    CovMatrix covmat;
    int n;
};
