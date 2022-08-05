
struct path_integral    {

    template<class T>
    class step_rep : public std::vector<std::pair<double, T>> {
        public:
        step_rep(size_t n, const T& initT) : std::vector<std::pair<double,T>>(n, std::make_pair(0,initT)) {}
        ~step_rep() {}

        void push(size_t i, double u, const T& t)   {
            (*this)[i] = std::pair<double,T>(u,t);
        }

        const T& val(size_t i) const    {
            return at(i).second;
        }

        double time(size_t i) const {
            return at(i).first;
        }
    };

    template<class T, class Lambda>
    static auto path_mean(const step_rep<T>& bp, Lambda lambda)   {
        double sum = (bp.time(1) - bp.time(0)) * lambda(bp.val(0));
        for (size_t i=1; i<bp.size(); i++)  {
            sum += (bp.time(i+1) - bp.time(i)) * lambda(bp.val(i));
        }
        return sum;
    }

    template<class T, class Lambda>
    static auto path_sum(const step_rep<T>& bp, double t_young, double t_old, Lambda lambda)   {
        return path_mean(bp, lambda) * (t_old - t_young);
    }

    template<class T1, class T2>
    static auto combine_rep(const step_rep<T1>& bp1, const step_rep<T2>& bp2)   {
        step_rep<std::pair<T1,T2>> bp(0, std::make_pair(bp1.val(0), bp2.val(0)));
        // ...
        return bp;
    }
};
