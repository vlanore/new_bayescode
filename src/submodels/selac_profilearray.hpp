
#pragma once

struct selac_profilearray {

    using T = std::vector<std::vector<double>>;

    using param_decl = param_decl_t<param<struct vector_argv, std::vector<double>>, param<struct real_arg, spos_real>>;

    static int aaindex(int i, int j) {
        return (i < j) ? (2 * Naa - i - 1) * i / 2 + j - i - 1
                       : (2 * Naa - j - 1) * j / 2 + i - j - 1;
    }

    static void gather(T& v, const std::vector<double>& aadist, spos_real psi) {
        for (size_t i=0; i<v.size(); i++)   {
            double tot = 0;
            for (size_t a=0; a<v.size(); a++)   {
                if (a == i) {
                    v[i][a] = 1.0;
                }
                else    {
                    double tmp = psi * aadist[aaindex(a,i)];
                    v[i][a] = 1e-8;
                    if (tmp < 100)  {
                        v[i][a] += exp(-tmp);
                    }
                }
                tot += v[i][a];
            }
            for (size_t a=0; a<v.size(); a++)   {
                v[i][a] /= tot;
                if (std::isnan(v[i][a]))    {
                    std::cerr << "selac profile: nan\n";
                    exit(1);
                }
            }
        }
    }
};

