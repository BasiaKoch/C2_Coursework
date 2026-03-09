#include "mphil_dis_cholesky.h"
#include <math.h>
#include <stdio.h>
#include <time.h>

#ifndef BLOCK_NB
#define BLOCK_NB 128   /* panel width; tune with -DBLOCK_NB=N */
#endif

#define MAX_N 100000

/*
 * v4_serial_blocked: serial panel-blocked Cholesky.
 *
 * Same panel-blocked algorithmic structure as v5_openmp_blocked, but without
 * OpenMP parallelism.
 *
 * Process nb columns at a time as a panel:
 *
 *   for k = 0, nb, 2*nb, ...
 *     1) factor the diagonal panel c[k:kend, k:kend]
 *     2) normalise the below-panel strip c[kend:n, k:kend]
 *     3) update the trailing submatrix with a rank-nb update
 *
 * Data layout:
 *   lower triangle (i >= j): c[i*n+j] = L[i,j]
 *   upper triangle (i < j): scratch space, not part of final output
 */

double mphil_dis_cholesky(double *c, int n)
{
    if (n < 1 || n > MAX_N) {
        fprintf(stderr, "mphil_dis_cholesky: n=%d out of range [1, %d]\n", n, MAX_N);
        return -1.0;
    }

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    for (int k = 0; k < n; k += BLOCK_NB) {
        int kend = (k + BLOCK_NB < n) ? (k + BLOCK_NB) : n;

        /* Panel factorisation: unblocked Cholesky on diagonal block */
        for (int p = k; p < kend; p++) {
            double *row_p = &c[(size_t)p * n];

            /* Diagonal pivot */
            double diag = sqrt(row_p[p]);
            row_p[p] = diag;
            double inv_diag = 1.0 / diag;

            /* Row normalisation within panel only */
            for (int j = p + 1; j < kend; j++) {
                row_p[j] *= inv_diag;
            }

            /* Column normalisation for all rows below diagonal */
            for (int i = p + 1; i < n; i++) {
                c[(size_t)i * n + p] *= inv_diag;
            }

            /* Within-panel trailing update */
            for (int i = p + 1; i < kend; i++) {
                double *row_i = &c[(size_t)i * n];
                double c_ip = row_i[p];
                for (int j = p + 1; j < kend; j++) {
                    row_i[j] -= c_ip * row_p[j];
                }
            }
        }

        /* Last panel: no trailing matrix left to update */
        if (kend >= n) {
            continue;
        }

        /* Serial trailing SYRK-style update of lower triangle */
        for (int i = kend; i < n; i++) {
            double *panel_i = &c[(size_t)i * n + k];
            double *row_i   = &c[(size_t)i * n];
            int panel_width = kend - k;

            for (int j = kend; j <= i; j++) {
                double *panel_j = &c[(size_t)j * n + k];
                double dot = 0.0;

                for (int p = 0; p < panel_width; p++) {
                    dot += panel_i[p] * panel_j[p];
                }

                row_i[j] -= dot;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    return (t_end.tv_sec  - t_start.tv_sec)
         + (t_end.tv_nsec - t_start.tv_nsec) * 1.0e-9;
}#include "mphil_dis_cholesky.h"
#include <math.h>
#include <stdio.h>
#include <time.h>

#ifndef BLOCK_NB
#define BLOCK_NB 128   /* panel width; tune with -DBLOCK_NB=N */
#endif

#define MAX_N 100000

/*
 * v4_serial_blocked: serial panel-blocked Cholesky.
 *
 * Same panel-blocked algorithmic structure as v5_openmp_blocked, but without
 * OpenMP parallelism.
 *
 * Process nb columns at a time as a panel:
 *
 *   for k = 0, nb, 2*nb, ...
 *     1) factor the diagonal panel c[k:kend, k:kend]
 *     2) normalise the below-panel strip c[kend:n, k:kend]
 *     3) update the trailing submatrix with a rank-nb update
 *
 * Data layout:
 *   lower triangle (i >= j): c[i*n+j] = L[i,j]
 *   upper triangle (i < j): scratch space, not part of final output
 */

double mphil_dis_cholesky(double *c, int n)
{
    if (n < 1 || n > MAX_N) {
        fprintf(stderr, "mphil_dis_cholesky: n=%d out of range [1, %d]\n", n, MAX_N);
        return -1.0;
    }

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    for (int k = 0; k < n; k += BLOCK_NB) {
        int kend = (k + BLOCK_NB < n) ? (k + BLOCK_NB) : n;

        /* Panel factorisation: unblocked Cholesky on diagonal block */
        for (int p = k; p < kend; p++) {
            double *row_p = &c[(size_t)p * n];

            /* Diagonal pivot */
            double diag = sqrt(row_p[p]);
            row_p[p] = diag;
            double inv_diag = 1.0 / diag;

            /* Row normalisation within panel only */
            for (int j = p + 1; j < kend; j++) {
                row_p[j] *= inv_diag;
            }

            /* Column normalisation for all rows below diagonal */
            for (int i = p + 1; i < n; i++) {
                c[(size_t)i * n + p] *= inv_diag;
            }

            /* Within-panel trailing update */
            for (int i = p + 1; i < kend; i++) {
                double *row_i = &c[(size_t)i * n];
                double c_ip = row_i[p];
                for (int j = p + 1; j < kend; j++) {
                    row_i[j] -= c_ip * row_p[j];
                }
            }
        }

        /* Last panel: no trailing matrix left to update */
        if (kend >= n) {
            continue;
        }

        /* Serial trailing SYRK-style update of lower triangle */
        for (int i = kend; i < n; i++) {
            double *panel_i = &c[(size_t)i * n + k];
            double *row_i   = &c[(size_t)i * n];
            int panel_width = kend - k;

            for (int j = kend; j <= i; j++) {
                double *panel_j = &c[(size_t)j * n + k];
                double dot = 0.0;

                for (int p = 0; p < panel_width; p++) {
                    dot += panel_i[p] * panel_j[p];
                }

                row_i[j] -= dot;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    return (t_end.tv_sec  - t_start.tv_sec)
         + (t_end.tv_nsec - t_start.tv_nsec) * 1.0e-9;
}