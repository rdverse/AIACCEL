#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include <time.h>

const int RUN_COUNT = 640000000;

// Single thread version
double calc_distance(double x, double y) {
    return x * x + y * y;
}

void single_thread_version() {
    int in_count = 0;  // points in and out of area
    int out_count = 0;

    for (int i = 0; i < RUN_COUNT; i++) {
        double x = (double)rand() / (double)RAND_MAX;
        double y = (double)rand() / (double)RAND_MAX;
        double dist = calc_distance(x, y);
        if (x * x + y * y <= 1) {
            in_count++;
        } else {
            out_count++;
        }
    }
    printf(
        "Single thread version\n"
        "In count: %d\n"
        "Out count: %d\n"
        "Pi: %f\n", in_count, out_count, 4 * (double)in_count / (in_count + out_count)
    );
}

// Multi thread version
struct monte_carlo_args {
    int N;
    double pi;
    unsigned int seed;
};

void *monte_carlo_sampling(void *data) {
    struct monte_carlo_args *p = (struct monte_carlo_args*)data;
    int in_count = 0;
    int out_count = 0;
    unsigned int seed = p->seed;

    for (int i = 0; i < p->N; i++) {
        double x = (double)rand_r(&seed) / (double)RAND_MAX;
        double y = (double)rand_r(&seed) / (double)RAND_MAX;
        double dist = calc_distance(x, y);
        if (x * x + y * y <= 1) {
            in_count++;
        } else {
            out_count++;
        }
    }
    p->pi = 4 * (double)in_count / (in_count + out_count);
    return NULL;
}

void multi_thread_version(int n_t) {
    int n_threads = n_t;
    pthread_t threads[n_threads];
    struct monte_carlo_args mca[n_threads];
    for (int i = 0; i < n_threads; i++) {
        mca[i].N = RUN_COUNT / n_threads;
        mca[i].pi = 0;
        mca[i].seed = rand(); // Seed each thread differently
        if (pthread_create(&threads[i], NULL, monte_carlo_sampling, &mca[i]) != 0) {
            perror("Failed to create thread");
            exit(1);
        }
    }

    for (int i = 0; i < n_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    double pi = 0;
    for (int i = 0; i < n_threads; i++) {
        pi += mca[i].pi;
    }
    printf(
        "Multi thread version\n"
        "Pi: %f\n", pi / n_threads
    );
}

int main() {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    single_thread_version();
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Time taken: %f seconds\n", time_taken);
    printf("\n");

    for (int i = 1; i <= 1000; i++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        multi_thread_version(i);
        clock_gettime(CLOCK_MONOTONIC, &end);
        time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("[n_threads %d] Time taken: %f seconds\n", i, time_taken);
        printf("\n");
    }
    return 0;
}