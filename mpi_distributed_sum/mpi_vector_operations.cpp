// Âåêòîðíûå îïåðàöèè â ðàñïðåäåë¸ííîé ïàìÿòè (MPI).

#include "mpi.h"
#include <iostream>
#include <vector>
#include <cstdlib>


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time = 0.0, end_time = 0.0;

    std::vector<double> A;     // ˜˜˜˜˜˜ ˜˜˜˜˜˜˜ ˜˜˜˜˜˜ ˜ rank 0
    std::vector<double> x(N);  // ˜˜˜˜˜˜ ˜˜˜˜˜˜ ˜ ˜˜˜˜
    std::vector<double> y(N);  // ˜˜˜˜˜˜ ˜˜˜˜˜˜˜˜˜ ˜ rank 0

    // ˜˜˜˜˜˜˜˜ ˜˜˜˜˜˜˜
    MPI_Bcast(x.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // ============ ˜˜˜˜˜˜˜˜˜˜ Scatterv ============

    std::vector<int> sendcounts(size); // ˜˜˜˜˜˜˜ ˜˜˜˜˜˜˜˜˜ ˜˜˜˜˜˜˜˜˜˜
    std::vector<int> displs(size);     // ˜˜˜˜˜˜˜˜

    int rows_per_proc = N / size;
    int extra = N % size;

    for (int p = 0; p < size; p++) {
        int rows = rows_per_proc + (p < extra ? 1 : 0);
        sendcounts[p] = rows * N;
        displs[p] = (p * rows_per_proc + std::min(p, extra)) * N;
    }

    // ˜˜˜˜˜˜˜˜˜ ˜˜˜˜˜˜
    int local_rows = sendcounts[rank] / N;
    std::vector<double> local_A(local_rows * N);
    std::vector<double> local_y(local_rows);

    // ˜˜˜˜˜˜˜˜ ˜˜˜˜˜˜˜˜˜
    MPI_Scatterv(
        A.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
        local_A.data(), sendcounts[rank], MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    // ˜˜˜˜˜˜˜˜˜˜˜˜˜ ˜˜˜˜˜ ˜˜˜˜˜˜˜˜˜˜ ˜˜˜˜˜˜˜
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // ---------- ˜˜˜˜˜˜˜˜˜ ˜˜˜˜˜˜˜˜˜ ----------
    for (int i = 0; i < local_rows; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            sum += local_A[i * N + j] * x[j];
        }
        local_y[i] = sum;
    }

    // ============ ˜˜˜˜ ˜˜˜˜˜˜˜˜˜˜ ˜˜˜˜˜ MPI_Gatherv ============

    std::vector<int> recvcounts(size);
    std::vector<int> recvdispls(size);

    for (int p = 0; p < size; p++) {
        int rows = rows_per_proc + (p < extra ? 1 : 0);
        recvcounts[p] = rows;
        recvdispls[p] = p * rows_per_proc + std::min(p, extra);
    }

    MPI_Gatherv(
        local_y.data(), local_rows, MPI_DOUBLE,
        y.data(), recvcounts.data(), recvdispls.data(), MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    if (rank == 0) {
        std::cout << "Processes: " << size
            << ", Time: " << (end_time - start_time) << " sec\n";
    }

    MPI_Finalize();
    return 0;
}