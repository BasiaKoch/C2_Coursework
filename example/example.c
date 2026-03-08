#include "mphil_dis_cholesky.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static double corr(double x, double y, double s)
{
    return 0.99 * exp(-0.5 * 16.0 * (x - y) * (x - y) / s / s);
}

int main(int argc, char *argv[])
{
    int n = 1000;
    if (argc > 1) n = atoi(argv[1]);
    if (n < 1 || n > 100000) {
        fprintf(stderr, "Usage: %s [n]  (1 <= n <= 100000)\n", argv[0]);
        return 1;
    }

    double *c = malloc((size_t)n * n * sizeof(double));
    if (!c) { fprintf(stderr, "malloc failed\n"); return 1; }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            c[n*i + j] = corr(i, j, n);
        c[n*i + i] = 1.0;
    }

    double elapsed = mphil_dis_cholesky(c, n);
    if (elapsed < 0.0) { free(c); return 1; }

    double logdet = 0.0;
    for (int p = 0; p < n; p++)
        logdet += log(c[p*n + p]);
    logdet *= 2.0;

    double gflops = (double)n * n * n / 3.0 / elapsed / 1.0e9;

    printf("n=%d  time=%.4f s  log|C|=%.6f  GFlop/s=%.4f\n",
           n, elapsed, logdet, gflops);

    free(c);
    return 0;
}
