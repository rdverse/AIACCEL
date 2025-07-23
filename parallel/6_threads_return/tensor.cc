#include <pthread.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cassert>

// Row-wise tensor parallelism: B = A * W + C
// A: MxN, W: NxP, C: MxP, B: MxP
// Partition output columns among T threads
// Cache-blocked inner multiplication for locality

struct ThreadArg {
    int tid;
    int T;
    int M, N, P;
    float *A;
    float *W;
    float *C;
    float *B;
    int start_col, end_col;
    pthread_barrier_t *barrier;
};

// Tile sizes for blocking (tunable)
#define TILE_M 32
#define TILE_N 32

void *thread_func(void *arg) {
    ThreadArg *t = (ThreadArg*)arg;
    int M = t->M, N = t->N;
    int sc = t->start_col, ec = t->end_col;
    float *A = t->A, *W = t->W, *C = t->C, *B = t->B;

    // Initialize B segment to C
    for (int j = sc; j < ec; ++j) {
        for (int i = 0; i < M; ++i) {
            B[i*P + j] = C[i*P + j];
        }
    }

    // Cache-blocked multiplication
    for (int mb = 0; mb < M; mb += TILE_M) {
        int mb_end = mb + TILE_M < M ? mb + TILE_M : M;
        for (int nb = 0; nb < N; nb += TILE_N) {
            int nb_end = nb + TILE_N < N ? nb + TILE_N : N;
            for (int j = sc; j < ec; ++j) {
                for (int k = nb; k < nb_end; ++k) {
                    float wkj = W[k*P + j];
                    float *Ai = &A[mb*N + k];
                    float *Bij = &B[mb*P + j];
                    for (int i = mb; i < mb_end; ++i) {
                        Bij[(i-mb)*P] += Ai[(i-mb)*N] * wkj;
                    }
                }
            }
        }
    }

    // Wait for all threads
    pthread_barrier_wait(t->barrier);
    return nullptr;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        printf("Usage: %s M N P T\n", argv[0]);
        return 1;
    }
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int P = atoi(argv[3]);
    int T = atoi(argv[4]);
    
    // Allocate
    float *A = (float*)aligned_alloc(64, M*N*sizeof(float));
    float *W = (float*)aligned_alloc(64, N*P*sizeof(float));
    float *C = (float*)aligned_alloc(64, M*P*sizeof(float));
    float *B = (float*)aligned_alloc(64, M*P*sizeof(float));
    assert(A && W && C && B);

    // Initialize with random data
    for (int i = 0; i < M*N; ++i) A[i] = drand48();
    for (int i = 0; i < N*P; ++i) W[i] = drand48();
    for (int i = 0; i < M*P; ++i) C[i] = drand48();

    pthread_t threads[T];
    ThreadArg args[T];
    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, nullptr, T);

    int per = P / T;
    for (int t = 0; t < T; ++t) {
        args[t].tid = t;
        args[t].T = T;
        args[t].M = M;
        args[t].N = N;
        args[t].P = P;
        args[t].A = A;
        args[t].W = W;
        args[t].C = C;
        args[t].B = B;
        args[t].start_col = t * per;
        args[t].end_col = (t+1 == T) ? P : (t+1)*per;
        args[t].barrier = &barrier;
        pthread_create(&threads[t], nullptr, thread_func, &args[t]);
    }
    for (int t = 0; t < T; ++t) {
        pthread_join(threads[t], nullptr);
    }

    // Optionally verify or print a checksum
    double sum = 0;
    for (int i = 0; i < M*P; ++i) sum += B[i];
    printf("Checksum: %f\n", sum);

    pthread_barrier_destroy(&barrier);
    free(A); free(W); free(C); free(B);
    return 0;
}

