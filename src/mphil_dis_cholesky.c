/*
 * mphil_dis_cholesky.c
 *
 * Baseline serial implementation of Cholesky factorization.
 * Version: v0.1-baseline
 *
 * This is a correct, readable, unoptimized implementation.
 * The loop structure is taken directly from the coursework specification.
 * Optimization is intentionally deferred to later stages (v0.2 onward)
 * following standard HPC practice: correctness first, speed second.
 *
 * Algorithm: right-looking Cholesky, column-by-column.
 *   Complexity: O(n^3 / 3) floating-point operations.
 *
 * Memory layout: row-major.  Element C[i][j] is at c[i*n + j].
 */

#include "mphil_dis_cholesky.h"

#include <math.h>    /* sqrt(), log() */
#include <stdio.h>   /* fprintf() */
#include <time.h>    /* clock_gettime() */

/* Maximum matrix dimension allowed, as stated in the spec. */
#define MAX_N 100000

double mphil_dis_cholesky(double *c, int n)
{
    /* ------------------------------------------------------------------
     * Bounds check required by the spec.
     * n must be a positive integer no greater than 100000.
     * ------------------------------------------------------------------ */
    if (n < 1 || n > MAX_N) {
        fprintf(stderr,
                "mphil_dis_cholesky: n = %d is outside the allowed range "
                "[1, %d]\n", n, MAX_N);
        return -1.0;
    }

    /* ------------------------------------------------------------------
     * Start wall-clock timer.
     * CLOCK_MONOTONIC gives elapsed real time and is not affected by
     * system clock adjustments.
     * The timer wraps only the computation, not setup or I/O.
     * ------------------------------------------------------------------ */
    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    /* ==================================================================
     * Cholesky factorization: right-looking, column-by-column.
     *
     * We march along the diagonal, processing one column panel per step.
     * At step p we use the already-computed columns 0..p-1 to update
     * column p and the trailing submatrix.
     *
     * After step p, the first p+1 columns of L are complete.
     * ================================================================== */
    for (int p = 0; p < n; p++) {

        /* --------------------------------------------------------------
         * Step 1 — Diagonal element.
         * Compute L[p][p] = sqrt(C[p][p]).
         * The input C[p][p] has already been reduced by previous steps.
         * -------------------------------------------------------------- */
        double diag = sqrt(c[p * n + p]);
        c[p * n + p] = diag;

        /* --------------------------------------------------------------
         * Step 2 — Row update (to the right of the diagonal).
         * Divide every element in row p to the right of the diagonal
         * by L[p][p].  These become the L^T entries (upper triangle).
         * -------------------------------------------------------------- */
        for (int j = p + 1; j < n; j++) {
            c[p * n + j] /= diag;
        }

        /* --------------------------------------------------------------
         * Step 3 — Column update (below the diagonal).
         * Divide every element in column p below the diagonal by L[p][p].
         * These become the L entries (lower triangle).
         * -------------------------------------------------------------- */
        for (int i = p + 1; i < n; i++) {
            c[i * n + p] /= diag;
        }

        /* --------------------------------------------------------------
         * Step 4 — Trailing submatrix rank-1 update.
         * Subtract the outer product of column p and row p from the
         * bottom-right submatrix.  This is the dominant O(n^2) work
         * per step, O(n^3/3) total.
         *
         * NOTE (baseline): the loop order here is outer=j, inner=i,
         * which accesses c[i*n+j] with stride n in the inner loop
         * (column-wise access in row-major storage → cache unfriendly).
         * Fixing this is the first serial optimisation in v0.2.
         * -------------------------------------------------------------- */
        for (int j = p + 1; j < n; j++) {
            for (int i = p + 1; i < n; i++) {
                c[i * n + j] -= c[i * n + p] * c[p * n + j];
            }
        }

    } /* end for p */

    /* ------------------------------------------------------------------
     * Stop wall-clock timer and return elapsed time in seconds.
     * ------------------------------------------------------------------ */
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double elapsed = (t_end.tv_sec  - t_start.tv_sec)
                   + (t_end.tv_nsec - t_start.tv_nsec) * 1.0e-9;
    return elapsed;
}
