#include "mphil_dis_cholesky.h"
#include <math.h>
#include <stdio.h>
#include <time.h>

#ifndef BLOCK_NB
#define BLOCK_NB 128   /* panel width; tune with -DBLOCK_NB=N */
#endif

#define MAX_N 100000

/*
 * v5_serial_blocked: serial panel-blocked Cholesky.
 *
 * Same panel-blocked algorithmic structure as v5_openmp_blocked, but without
 * any OpenMP directives.  Serves as the serial reference for the blocked
 * algorithm so that the speedup from OpenMP parallelism can be separated
 * from the speedup due to cache-friendlier blocking.
 *
 * Why blocking helps even in serial
 * ──────────────────────────────────
 * The unblocked algorithm (v2/v3) applies one rank-1 update per step p,
 * reading the entire row p and each row i > p on every step.  For large n
 * the matrix does not fit in L3 cache and each step causes many cache misses.
 *
 * Processing nb columns at a time (a "panel") amortises the memory traffic:
 * the panel strip c[i, k:kend] (nb doubles = nb×8 bytes) stays in L1/L2
 * while the inner dot-product loop runs.  The dominant O(n²·nb) trailing
 * update is therefore mostly cache-hit, increasing arithmetic intensity.
 *
 * Algorithm
 * ─────────
 *   for k = 0, nb, 2·nb, ...
 *     1) Panel factorisation: unblocked Cholesky on c[k:kend, k:kend] and
 *        column-normalise the below-panel strip c[kend:n, k:kend].
 *     2) Trailing SYRK: for each row i ≥ kend subtract the rank-nb outer
 *        product c[i, k:kend] · c[j, k:kend] from c[i, j], j in [kend, i].
 *     3) Fill upper triangle from lower (spec: c[i,j] = L^T[i,j] for i < j).
 *
 * Data layout:
 *   lower triangle (i ≥ j): c[i*n+j] = L[i,j]
 *   upper triangle (i < j): c[i*n+j] = L^T[i,j] = L[j,i]  (filled at end)
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

        /* ── Panel factorisation ──────────────────────────────────────────
         * Unblocked right-looking Cholesky on the diagonal block c[k:kend,
         * k:kend].  Column normalisation covers ALL rows below diagonal
         * (including rows kend..n-1), so L[i,p] is ready for the SYRK. */
        for (int p = k; p < kend; p++) {
            double *row_p = &c[(size_t)p * n];

            double diag     = sqrt(row_p[p]);
            row_p[p]        = diag;
            double inv_diag = 1.0 / diag;

            /* Row normalisation: within-panel columns only (p+1..kend-1).
             * Columns j ≥ kend are set by SYRK; skipping them avoids
             * reading cold cache lines outside the current panel. */
            for (int j = p + 1; j < kend; j++)
                row_p[j] *= inv_diag;

            /* Column normalisation for all rows p+1..n-1.
             * This is the TRSM step: sets c[i,p] = L[i,p] for rows below
             * the panel so the SYRK can use them without further division. */
            for (int i = p + 1; i < n; i++)
                c[(size_t)i * n + p] *= inv_diag;

            /* Within-panel trailing update (rows and cols p+1..kend-1).
             * Below-panel rows are deferred to the SYRK. */
            for (int i = p + 1; i < kend; i++) {
                double *row_i = &c[(size_t)i * n];
                double  c_ip  = row_i[p];
                for (int j = p + 1; j < kend; j++)
                    row_i[j] -= c_ip * row_p[j];
            }
        }

        if (kend >= n) continue;  /* last panel: no trailing submatrix */

        /* ── Trailing SYRK ────────────────────────────────────────────────
         * Subtract the rank-nb outer product from the lower triangle of the
         * trailing submatrix:
         *
         *   c[i, j] -= Σ_{p=k}^{kend-1}  c[i,p] · c[j,p]   j ∈ [kend, i]
         *
         * Both panel strips are stride-1 (nb doubles), so the inner p-loop
         * is cache-friendly and auto-vectorises with -ffast-math. */
        for (int i = kend; i < n; i++) {
            double *panel_i     = &c[(size_t)i * n + k];
            double *row_i       = &c[(size_t)i * n];
            int     panel_width = kend - k;

            for (int j = kend; j <= i; j++) {
                double *panel_j = &c[(size_t)j * n + k];
                double  dot     = 0.0;
                for (int p = 0; p < panel_width; p++)
                    dot += panel_i[p] * panel_j[p];
                row_i[j] -= dot;
            }
        }
    }

    /* ── Fill upper triangle ──────────────────────────────────────────────
     * Spec requires c[i,j] = L^T[i,j] = L[j,i] for i < j.
     * Panel row-normalisation only covers j < kend; copy lower → upper. */
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            c[i * n + j] = c[j * n + i];

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    return (t_end.tv_sec  - t_start.tv_sec)
         + (t_end.tv_nsec - t_start.tv_nsec) * 1.0e-9;
}
