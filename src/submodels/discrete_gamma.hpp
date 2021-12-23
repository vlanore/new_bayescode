
#include "incomplete_gamma.hpp"

struct discrete_gamma {
    using T = std::vector<double>;
    using param_decl = param_decl_t<param<struct shape, spos_real>>;

    static void gather(T& v, spos_real alpha)    {
        size_t ncat = v.size();
        vector<double> x(ncat,0);
        vector<double> y(ncat,0);
        double lg = LnGamma(alpha+1.0);
        for (size_t i=0; i<ncat; i++)  {
            x[i] = PointGamma((i+1.0)/ncat,alpha,alpha);
        }
        for (size_t i=0; i<ncat-1; i++)	{
            y[i] = IncompleteGamma(alpha*x[i],alpha+1,lg);
        }
        y[ncat-1] = 1.0;
        v[0] = ncat * y[0];
        for (size_t i=1; i<ncat; i++)	{
            v[i] = ncat * (y[i] - y[i-1]);
        }
    }
};


