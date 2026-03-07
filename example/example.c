/*
 * example.c
 *
 * Example program demonstrating use of mphil_dis_cholesky.
 *
 * Generates a symmetric positive-definite n x n matrix using the
 * corr() function from the coursework spec, factorizes it, and reports:
 *   - wall-clock time returned by the library
 *   - log-determinant computed from the diagonal of L
 *   - GFlop/s achieved  (Cholesky ~ n^3/3 operations)
 *
 * Usage:
 *   ./example          uses n = 1000 (default)
 *   ./example 2000     uses n = 2000
 *
 * Build:  make example
 * Run:    ./example/example [n]
 */

#include "mphil_dis_cholesky.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* ------------------------------------------------------------------
 * corr() — correlation function from the coursework spec.
 * Generates a decaying Gaussian correlation between indices x and y
 * in a matrix of size s.  The diagonal is set to 1.0 afterwards so
 * the matrix is strictly positive-definite.
 * ------------------------------------------------------------------ */
static double corr(double x, double y, double s)
{
    return 0.99 * exp(-0.5 * 16.0 * (x - y) * (x - y) / s / s);
}

int main(int argc, char *argv[])
{
    /* Parse optional size argument */
    int n = 1000;
    if (argc > 1) {
        n = atoi(argv[1]);
        if (n < 1 || n > 100000) {
            fprintf(stderr, "Usage: %s [n]   where 1 <= n <= 100000\n",
                    argv[0]);
            return 1;
        }
    }

    printf("mphil_dis_cholesky example  (v0.1-baseline)\n");
    printf("Matrix size:  n = %d\n", n);
    printf("Memory:       %.1f MB\n",
           (double)n * n * sizeof(double) / (1024.0 * 1024.0));

    /* Allocate matrix */
    double *c = malloc((size_t)n * n * sizeof(double));
    if (!c) {
        fprintf(stderr, "malloc failed for n=%d\n", n);
        return 1;
    }

    /* Fill with corr() positive-definite matrix (from spec) */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            c[n * i + j] = corr(i, j, n);
        c[n * i + i] = 1.0;   /* ensure diagonal dominance */
    }

    /* Factorize — the function modifies c in-place and returns timing */
    double elapsed = mphil_dis_cholesky(c, n);

    if (elapsed < 0.0) {
        fprintf(stderr, "Factorization failed.\n");
        free(c);
        return 1;
    }

    /* Compute log|C| = 2 * sum_p  log(L[p,p])  (Eq. 4 from spec) */
    double logdet = 0.0;
    for (int p = 0; p < n; p++)
        logdet += log(c[p * n + p]);
    logdet *= 2.0;

    /* Cholesky operation count is approximately n^3 / 3 */
    double gflops = (double)n * (double)n * (double)n / 3.0 / elapsed / 1.0e9;

    printf("Elapsed time: %.6f s\n", elapsed);
    printf("log|C|:       %.10f\n", logdet);
    printf("GFlop/s:      %.4f\n", gflops);

    free(c);
    return 0;
}
