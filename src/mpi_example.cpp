#include "mpi_components/gather.hpp"
using namespace std;

void compute(int, char**) {
    auto rank = MPI::p->rank;  // getting MPI rank
    auto size = MPI::p->size;  // ... and size

    // List of indices for data partition
    // indices are partitioned into size-1 bins, starting at process 1
    IndexSet indices{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"};
    Partition partition(indices, size - 1, 1);

    // Actual data structure
    vector<double> data;
    if (!rank) {  // master
        data = vector<double>(indices.size(), 0);
    } else {  // slave
        data = vector<double>(partition.my_partition_size(), rank);
    }

    // Creating gather operation (nor executed yet)
    auto g = gather(partition, data);

    // Executing gather operation
    slave_to_master(g);

    // Displaying data per process
    MPI::p->message("data: {}", vector_to_string(data));
}

int main(int argc, char** argv) { mpi_run(argc, argv, compute); }