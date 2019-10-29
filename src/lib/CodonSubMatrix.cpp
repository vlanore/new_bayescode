#include "CodonSubMatrix.hpp"
using namespace std;

void MGCodonSubMatrix::ComputeArray(int i) const {
    double total = 0;
    for (int j = 0; j < GetNstate(); j++) {
        if (i != j) {
            int pos = GetDifferingPosition(i, j);
            if ((pos != -1) && (pos != 3)) {
                int a = GetCodonPosition(pos, i);
                int b = GetCodonPosition(pos, j);
                Q(i, j) = (*NucMatrix)(a, b);
                total += Q(i, j);
            } else {
                Q(i, j) = 0;
            }
        }
    }
    Q(i, i) = -total;
}

void MGCodonSubMatrix::ComputeStationary() const {
    if (! GetCodonStateSpace())    {
        std::cerr << "in MG compute stat: null statespace\n";
        exit(1);
    }
    if (! NucMatrix)    {
        std::cerr << "in MG compute stat: null nuc matrix\n";
        exit(1);
    }
    // compute stationary probabilities
    double total = 0;
    for (int i = 0; i < GetNstate(); i++) {
        mStationary[i] = NucMatrix->Stationary(GetCodonPosition(0, i)) *
                         NucMatrix->Stationary(GetCodonPosition(1, i)) *
                         NucMatrix->Stationary(GetCodonPosition(2, i));
        total += mStationary[i];
    }

    // renormalize stationary probabilities
    for (int i = 0; i < GetNstate(); i++) { mStationary[i] /= total; }
}

void MGOmegaCodonSubMatrix::ComputeArray(int i) const {
    if (! GetCodonStateSpace())    {
        std::cerr << "in MG compute array: null statespace\n";
        exit(1);
    }
    if (! NucMatrix)    {
        std::cerr << "in MG compute array: null nuc matrix\n";
        exit(1);
    }
    double total = 0;
    for (int j = 0; j < GetNstate(); j++) {
        if (i != j) {
            int pos = GetDifferingPosition(i, j);
            if ((pos != -1) && (pos != 3)) {
                int a = GetCodonPosition(pos, i);
                int b = GetCodonPosition(pos, j);
                if (a == b) {
                    cerr << GetCodonStateSpace()->GetState(i) << '\t'
                         << GetCodonStateSpace()->GetState(j) << '\n';
                    cerr << pos << '\n';
                    exit(1);
                }
                Q(i, j) = (*NucMatrix)(a, b);
                if (!Synonymous(i, j)) { Q(i, j) *= GetOmega(); }
            } else {
                Q(i, j) = 0;
            }
            total += Q(i, j);
        }
    }
    Q(i, i) = -total;
    if (total < 0) {
        cerr << "negative rate away\n";
        exit(1);
    }
}

