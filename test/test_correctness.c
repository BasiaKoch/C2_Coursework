/*
 * test_correctness.c
 *
 * Correctness tests for mphil_dis_cholesky (v0.1-baseline).
 *
 * Tests performed:
 *   1. 2x2 spec example — exact known answer from the coursework handout.
 *   2. 3x3 hand-computed example — exact L verified by hand.
 *   3. L * L^T reconstruction — factorise a matrix, multiply L back,
 *      check we recover the original C to floating-point precision.
 *   4. Log-determinant formula — verify  log|C| = 2 * sum(log(L[p,p])).
 *   5. Bounds check — n=0 and n>100000 must return -1.0.
 *
 * Build and run:   make test
 *
 * The program exits with status 0 if all tests pass, 1 otherwise.
 */

#include "mphil_dis_cholesky.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Floating-point tolerance for element-wise comparisons. */
#define TOL 1e-10

/* ------------------------------------------------------------------ */
/* Minimal test framework                                              */
/* ------------------------------------------------------------------ */

static int g_tests_run    = 0;
static int g_tests_failed = 0;

static void check(const char *label, int condition)
{
    g_tests_run++;
    if (condition) {
        printf("  PASS  %s\n", label);
    } else {
        printf("  FAIL  %s\n", label);
        g_tests_failed++;
    }
}

/* ------------------------------------------------------------------ */
/* Test 1: 2x2 example from the spec                                  */
/*                                                                     */
/* C2 = | 4   2  |     Expected L2 = | 2  0 |                        */
/*      | 2  26  |                   | 1  5 |                         */
/*                                                                     */
/* After in-place factorization the array should hold:                 */
/*   { 2, 1, 1, 5 }                                                   */
/* (lower triangle = L, upper triangle = L^T)                         */
/* ------------------------------------------------------------------ */
static void test_2x2_spec(void)
{
    printf("\n=== Test 1: 2x2 spec example ===\n");

    double c[4] = {4.0, 2.0,
                   2.0, 26.0};

    double t = mphil_dis_cholesky(c, 2);

    printf("  Result array: {%.6f, %.6f, %.6f, %.6f}\n",
           c[0], c[1], c[2], c[3]);
    printf("  Elapsed: %.6e s\n", t);

    /* L (lower triangle + diagonal) */
    check("L[0,0] = 2.0",           fabs(c[0*2+0] - 2.0) < TOL);
    check("L[1,0] = 1.0",           fabs(c[1*2+0] - 1.0) < TOL);
    check("L[1,1] = 5.0",           fabs(c[1*2+1] - 5.0) < TOL);

    /* L^T (upper triangle must mirror L) */
    check("L^T[0,1] = L[1,0] = 1.0", fabs(c[0*2+1] - 1.0) < TOL);

    /* Log-determinant: det(C2) = 4*26 - 2*2 = 100 */
    double logdet = 2.0 * (log(c[0*2+0]) + log(c[1*2+1]));
    printf("  log|C| = %.10f  (expected log(100) = %.10f)\n",
           logdet, log(100.0));
    check("log|C| = log(100)",       fabs(logdet - log(100.0)) < TOL);

    check("return value >= 0",       t >= 0.0);
}

/* ------------------------------------------------------------------ */
/* Test 2: 3x3 hand-computed example                                  */
/*                                                                     */
/* C3 = | 4  2  2 |                                                   */
/*      | 2  3  1 |                                                    */
/*      | 2  1  3 |                                                    */
/*                                                                     */
/* Hand-computed L3 (work shown below):                               */
/*                                                                     */
/*  p=0: diag = sqrt(4) = 2                                           */
/*       row:  C[0,1]/=2 -> 1,  C[0,2]/=2 -> 1                       */
/*       col:  C[1,0]/=2 -> 1,  C[2,0]/=2 -> 1                       */
/*       submatrix update:                                             */
/*         C[1,1] -= C[1,0]*C[0,1] = 3 - 1*1 = 2                    */
/*         C[1,2] -= C[1,0]*C[0,2] = 1 - 1*1 = 0                    */
/*         C[2,1] -= C[2,0]*C[0,1] = 1 - 1*1 = 0                    */
/*         C[2,2] -= C[2,0]*C[0,2] = 3 - 1*1 = 2                    */
/*                                                                     */
/*  p=1: diag = sqrt(2)                                               */
/*       row:  C[1,2]/=sqrt(2) -> 0                                   */
/*       col:  C[2,1]/=sqrt(2) -> 0                                   */
/*       submatrix update:  C[2,2] -= 0*0 = 2  (unchanged)           */
/*                                                                     */
/*  p=2: diag = sqrt(2)                                               */
/*                                                                     */
/* Result L3 = | 2       0        0      |                            */
/*             | 1    sqrt(2)     0      |                            */
/*             | 1       0      sqrt(2) |                             */
/*                                                                     */
/* det(C3) = 16  ->  log|C3| = 2*(log2 + log(sqrt2) + log(sqrt2))   */
/*                            = 2*2*log2 = log(16)                    */
/* ------------------------------------------------------------------ */
static void test_3x3_hand(void)
{
    printf("\n=== Test 2: 3x3 hand-computed example ===\n");

    double c[9] = {4.0, 2.0, 2.0,
                   2.0, 3.0, 1.0,
                   2.0, 1.0, 3.0};

    mphil_dis_cholesky(c, 3);

    printf("  Result matrix:\n");
    for (int i = 0; i < 3; i++)
        printf("    [%10.6f  %10.6f  %10.6f]\n",
               c[i*3+0], c[i*3+1], c[i*3+2]);

    double sq2 = sqrt(2.0);

    check("L[0,0] = 2.0",      fabs(c[0*3+0] - 2.0) < TOL);
    check("L[1,0] = 1.0",      fabs(c[1*3+0] - 1.0) < TOL);
    check("L[1,1] = sqrt(2)",  fabs(c[1*3+1] - sq2)  < TOL);
    check("L[2,0] = 1.0",      fabs(c[2*3+0] - 1.0) < TOL);
    check("L[2,1] = 0.0",      fabs(c[2*3+1] - 0.0) < TOL);
    check("L[2,2] = sqrt(2)",  fabs(c[2*3+2] - sq2)  < TOL);

    double logdet = 2.0 * (log(c[0*3+0]) + log(c[1*3+1]) + log(c[2*3+2]));
    printf("  log|C| = %.10f  (expected log(16) = %.10f)\n",
           logdet, log(16.0));
    check("log|C| = log(16)",  fabs(logdet - log(16.0)) < TOL);
}

/* ------------------------------------------------------------------ */
/* Helpers for Test 3                                                  */
/* ------------------------------------------------------------------ */

/* corr() matrix generator from the spec. */
static double corr(double x, double y, double s)
{
    return 0.99 * exp(-0.5 * 16.0 * (x - y) * (x - y) / s / s);
}

static void fill_corr(double *c, int n)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            c[i * n + j] = corr(i, j, n);
        c[i * n + i] = 1.0;
    }
}

/* ------------------------------------------------------------------ */
/* Test 3: L * L^T reconstruction                                     */
/*                                                                     */
/* Factorise a corr() matrix, then compute L * L^T by hand and        */
/* check that every element matches the original C.                   */
/*                                                                     */
/* L * L^T [i,j] = sum_{k=0}^{min(i,j)} L[i,k] * L[j,k]             */
/* (L is lower triangular, so L[i,k] = 0 for k > i)                 */
/* ------------------------------------------------------------------ */
static void test_reconstruction(int n)
{
    printf("\n=== Test 3: L*L^T reconstruction  (n=%d) ===\n", n);

    double *c_orig = malloc((size_t)n * n * sizeof(double));
    double *c_fact = malloc((size_t)n * n * sizeof(double));

    if (!c_orig || !c_fact) {
        printf("  SKIP (malloc failed)\n");
        free(c_orig);
        free(c_fact);
        return;
    }

    fill_corr(c_orig, n);
    memcpy(c_fact, c_orig, (size_t)n * n * sizeof(double));

    double t = mphil_dis_cholesky(c_fact, n);
    printf("  Elapsed: %.6e s\n", t);

    /* Compute max element-wise error  |L*L^T - C_orig| */
    double max_err = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            /* Product element (i,j): sum over k from 0 to min(i,j) */
            int k_max = (i < j) ? i : j;
            double sum = 0.0;
            for (int k = 0; k <= k_max; k++)
                sum += c_fact[i * n + k] * c_fact[j * n + k];
            double err = fabs(sum - c_orig[i * n + j]);
            if (err > max_err) max_err = err;
        }
    }
    printf("  Max |L*L^T - C| = %.2e\n", max_err);

    char label[64];
    snprintf(label, sizeof(label), "max |L*L^T - C| < 1e-10  (got %.2e)", max_err);
    check(label, max_err < 1e-10);

    /* Print log|C| for reference */
    double logdet = 0.0;
    for (int p = 0; p < n; p++) logdet += log(c_fact[p * n + p]);
    logdet *= 2.0;
    printf("  log|C| = %.10f\n", logdet);

    free(c_orig);
    free(c_fact);
}

/* ------------------------------------------------------------------ */
/* Test 4: Bounds check                                                */
/* ------------------------------------------------------------------ */
static void test_bounds(void)
{
    printf("\n=== Test 4: Bounds check ===\n");

    double c[4] = {1.0, 0.0, 0.0, 1.0};
    double t;

    t = mphil_dis_cholesky(c, 0);
    check("n=0       returns -1.0", t == -1.0);

    t = mphil_dis_cholesky(c, 100001);
    check("n=100001  returns -1.0", t == -1.0);

    /* n=1: identity 1x1, sqrt(1.0) = 1.0 */
    double c1[1] = {4.0};
    t = mphil_dis_cholesky(c1, 1);
    check("n=1  returns >= 0",      t >= 0.0);
    check("n=1  c[0] = sqrt(4) = 2.0", fabs(c1[0] - 2.0) < TOL);
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */
int main(void)
{
    printf("=======================================================\n");
    printf("mphil_dis_cholesky correctness tests  (v0.1-baseline)\n");
    printf("=======================================================\n");

    test_2x2_spec();
    test_3x3_hand();
    test_reconstruction(5);
    test_reconstruction(50);
    test_reconstruction(200);
    test_bounds();

    printf("\n-------------------------------------------------------\n");
    if (g_tests_failed == 0)
        printf("ALL %d TESTS PASSED\n", g_tests_run);
    else
        printf("%d / %d TESTS FAILED\n", g_tests_failed, g_tests_run);
    printf("-------------------------------------------------------\n");

    return (g_tests_failed == 0) ? 0 : 1;
}
