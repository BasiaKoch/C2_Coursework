#include "mphil_dis_cholesky.h"
#include <math.h>
#include <stdio.h>
#include <time.h>

#define MAX_N 100000

/* BLOCK_SIZE: width of the j-tile in the trailing-submatrix update.
 * Keeping BS columns of row_p hot in L1 improves temporal reuse across i-rows.
 * Override at compile time: -DBLOCK_SIZE=32  (default 64). */
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif

double mphil_dis_cholesky(double *c, int n)
{
    if (n < 1 || n > MAX_N) {
        fprintf(stderr, "mphil_dis_cholesky: n=%d out of range [1, %d]\n", n, MAX_N);
        return -1.0;
    }

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    for (int p = 0; p < n; p++) {
        double *row_p = &c[p*n];

        double diag = sqrt(row_p[p]);
        row_p[p] = diag;

        double inv_diag = 1.0 / diag;

        for (int j = p + 1; j < n; j++)
            row_p[j] *= inv_diag;

        for (int i = p + 1; i < n; i++)
            c[i*n + p] *= inv_diag;

        /* Trailing submatrix update, blocked in j.
         * Each tile loads BLOCK_SIZE elements of row_p once and reuses them
         * across all (n-p-1) i-rows, keeping that slice in L1 cache. */
        for (int jj = p + 1; jj < n; jj += BLOCK_SIZE) {
            int jend = jj + BLOCK_SIZE < n ? jj + BLOCK_SIZE : n;

            for (int i = p + 1; i < n; i++) {
                double *row_i = &c[i*n];
                double c_ip = row_i[p];

                for (int j = jj; j < jend; j++)
                    row_i[j] -= c_ip * row_p[j];
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    return (t_end.tv_sec  - t_start.tv_sec)
         + (t_end.tv_nsec - t_start.tv_nsec) * 1.0e-9;
}
