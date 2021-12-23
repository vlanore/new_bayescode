
#pragma once

struct selac_profile {

    using T = std::vector<double>;

    using param_decl = param_decl_t<param<struct exchrates, std::vector<double>>, param<struct int_arg, integer>, param<struct intensity, spos_real>>;

    static int aaindex(int i, int j) {
        return (i < j) ? (2 * Naa - i - 1) * i / 2 + j - i - 1
                       : (2 * Naa - j - 1) * j / 2 + i - j - 1;
    }

    static void gather(T& v, integer i, std::vector<double>& aadist, spos_real psi) {
        double tot = 0;
        for (size_t a=0; a<v.size(); a++)   {
            if (a == i) {
                v[a] = 1.0;
            }
            else    {
                double tmp = psi * aadist[aaindex(a,i)];
                v[a] = 1e-8;
                if (tmp < 100)  {
                    v[a] += exp(-tmp);
                }
            }
            tot += v[a];
        }
        for (size_t a=0; a<v.size(); a++)   {
            v[a] /= tot;
        }
};

