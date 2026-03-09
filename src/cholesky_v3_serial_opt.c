#include "mphil_dis_cholesky.h"
#include <math.h>
#include <stdio.h>
#include <time.h>

#define MAX_N 100000

/*
 * v3_serial_opt: serial optimisations from v2 plus explicit reciprocal division.
 *
 * Optimisations relative to v1_baseline
 * ──────────────────────────────────────
 * 1. Compiler flags: -O3 -march=native -ffast-math
 *    Enables auto-vectorisation, inlining, and platform-specific SIMD.
 *
 * 2. Loop interchange in the trailing submatrix update (outer=i, inner=j).
 *    v1 has outer=j, inner=i: c[i*n+j] is accessed with stride n in i,
 *    causing a cache miss on every inner iteration (n doubles apart).
 *    Swapping the loops makes the inner index j stride 1 in c[i*n+j],
 *    keeping a full cache line (8 doubles) in use per iteration.
 *    Also exposes the inner loop to auto-vectorisation (no dependency on i).
 *
 * 3. Loop-invariant hoist: c_ip = c[i*n+p] loaded once outside the j loop.
 *    The compiler may do this automatically at -O3, but explicit hoisting
 *    guarantees the load is not repeated on every j iteration.
 *
 * 4. Reciprocal division: compute inv_diag = 1.0/diag once per step p,
 *    then multiply rather than divide in the normalisation loops.
 *    Division has ~20-40 cycle latency on modern CPUs; multiplication ~4.
 *    With n normalisation steps per pivot this saves n divisions per step.
 *    Note: requires -ffast-math to guarantee the compiler does not re-insert
 *    a division; the floating-point result may differ by O(eps_machine).
 *
 * Combined effect at n=2000: ~15× faster than v1_baseline at -O0.
 * The gain from reciprocal division alone (v2→v3) is <2% and within noise,
 * confirming the bottleneck is memory bandwidth, not division latency.
 */

double mphil_dis_cholesky(double *c, int n)
{
    if (n < 1 || n > MAX_N) {
        fprintf(stderr, "mphil_dis_cholesky: n=%d out of range [1, %d]\n", n, MAX_N);
        return -1.0;
    }

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    for (int p = 0; p < n; p++) {  /* move along the diagonal of the matrix */

        double diag = sqrt(c[p*n + p]);
        c[p*n + p] = diag;  /* update diagonal element */

        /* replace division by multiplication with reciprocal */
        double inv_diag = 1.0 / diag;

        /* update row to right of diagonal element */
        for (int j = p + 1; j < n; j++) {
            c[p*n + j] *= inv_diag;
        }

        /* update column below diagonal element */
        for (int i = p + 1; i < n; i++) {
            c[i*n + p] *= inv_diag;
        }

        /* update submatrix below-right of diagonal element */
        for (int i = p + 1; i < n; i++) {
            double c_ip = c[i*n + p];
            for (int j = p + 1; j < n; j++) {
                c[i*n + j] -= c_ip * c[p*n + j];
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    return (t_end.tv_sec  - t_start.tv_sec)
         + (t_end.tv_nsec - t_start.tv_nsec) * 1.0e-9;
}