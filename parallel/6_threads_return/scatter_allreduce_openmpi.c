#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <math.h>

int main(int argc, char** argv) {
    int rank, size;
    const long long N = 10000000; // 100 million elements
    int *data = NULL;
    long long chunk_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    chunk_size = N / size;
    int* local_chunk = (int*)malloc(chunk_size * sizeof(int));

    // Root process initializes the data
    if (rank == 0) {
        data = (int*)malloc(N * sizeof(int));
        for (long long i = 0; i < N; i++) {
            data[i] = i % 100000;
        }
    }

    // Scatter: distribute chunks to all processes
    MPI_Scatter(data, chunk_size, MPI_INT, local_chunk, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process does heavy computation on its chunk in parallel
    double local_sum = 0.0;
    #pragma omp parallel for reduction(+:local_sum)
    for (long long i = 0; i < chunk_size; i++) {
        double val = local_chunk[i];
        // Heavy computation: simulate work
        for (int j = 0; j < 1000; j++) {
            val = sin(val) * cos(val) + sqrt(val + 1.0)/1000;
        }
        local_sum += val;
    }

    // All-Reduce: sum all local sums and distribute result to all processes
    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    printf("Rank %d: local sum = %f, global sum = %f\n", rank, local_sum, global_sum);

    // Cleanup
    if (rank == 0) free(data);
    free(local_chunk);

    MPI_Finalize();
    return 0;
}