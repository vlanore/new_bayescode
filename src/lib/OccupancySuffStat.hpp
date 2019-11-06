#pragma once

/**
 * \brief A sufficient statistic for a multinomial mixture allocation vector
 *
 * Given a mixture of K components, and a vector of N integers,
 * specifying the allocation of N items to the component of the mixtures,
 * OccupancySuffStat simply stores the counts (numbers of sites allocated to
 * each component), which is a sufficient statistic for the underlying component
 * weights.
 *
 */

class OccupancySuffStat : public std::vector<size_t>    {
    
  public:
    //! \brief constructor (parameterized by mixture size)
    OccupancySuffStat(size_t insize) : std::vector<size_t>(insize, 0) {}
    ~OccupancySuffStat() {}

    //! reset count vector
    void Clear() {
        for (size_t i = 0; i < size(); i++) { (*this)[i] = 0; }
    }

    //! implement additive behavior of OccupancySuffStat
    void Add(const OccupancySuffStat &from) {
        assert(size() == from.size());
        for (size_t i = 0; i < size(); i++) { (*this)[i] += from[i]; }
    }

    //! implement additive behavior of OccupancySuffStat
    OccupancySuffStat &operator+=(const OccupancySuffStat &from) {
        Add(from);
        return *this;
    }

    //! add suff stat based on an allocation vector
    void AddSuffStat(const std::vector<size_t> &alloc) {
        for (size_t i = 0; i < alloc.size(); i++) { 
            assert (alloc[i] <(*this).size());
            (*this)[alloc[i]]++; 
        }
    }

    bool operator==(const OccupancySuffStat& other) const {
        if (size() != other.size()) {
            return false;
        }
        for (size_t i = 0; i < size(); i++) { 
            if ((*this)[i] != other[i]) {
                return false;
            }
        }
        return true;
    }
};

