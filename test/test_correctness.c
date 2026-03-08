#include "mphil_dis_cholesky.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TOL 1e-10

static int tests_run = 0, tests_failed = 0;

static void check(const char *label, int ok)
{
    tests_run++;
    if (ok) printf("  PASS  %s\n", label);
    else   { printf("  FAIL  %s\n", label); tests_failed++; }
}

/* --- Test 1: 2x2 spec example --- */
static void test_2x2(void)
{
    printf("\n=== Test 1: 2x2 spec example ===\n");
    double c[4] = {4.0, 2.0, 2.0, 26.0};
    double t = mphil_dis_cholesky(c, 2);

    check("L[0,0] = 2",           fabs(c[0] - 2.0) < TOL);
    check("L[1,0] = 1",           fabs(c[2] - 1.0) < TOL);
    check("L[1,1] = 5",           fabs(c[3] - 5.0) < TOL);
    check("L^T[0,1] = 1",         fabs(c[1] - 1.0) < TOL);
    check("log|C| = log(100)",     fabs(2*(log(c[0])+log(c[3])) - log(100.0)) < TOL);
    check("return >= 0",           t >= 0.0);
}

/* --- Test 2: 3x3 hand-computed ---
 * C = {{4,2,2},{2,3,1},{2,1,3}}, det=16
 * L = {{2,0,0},{1,sqrt(2),0},{1,0,sqrt(2)}}
 */
static void test_3x3(void)
{
    printf("\n=== Test 2: 3x3 hand-computed ===\n");
    double c[9] = {4.0,2.0,2.0, 2.0,3.0,1.0, 2.0,1.0,3.0};
    mphil_dis_cholesky(c, 3);

    double sq2 = sqrt(2.0);
    check("L[0,0]=2",      fabs(c[0] - 2.0) < TOL);
    check("L[1,0]=1",      fabs(c[3] - 1.0) < TOL);
    check("L[1,1]=sqrt2",  fabs(c[4] - sq2)  < TOL);
    check("L[2,0]=1",      fabs(c[6] - 1.0) < TOL);
    check("L[2,1]=0",      fabs(c[7] - 0.0) < TOL);
    check("L[2,2]=sqrt2",  fabs(c[8] - sq2)  < TOL);
    check("log|C|=log(16)", fabs(2*(log(c[0])+log(c[4])+log(c[8])) - log(16.0)) < TOL);
}

/* --- Test 3: L*L^T reconstruction with corr() matrix --- */
static double corr(double x, double y, double s)
{
    return 0.99 * exp(-0.5 * 16.0 * (x-y)*(x-y) / s / s);
}

static void test_reconstruction(int n)
{
    printf("\n=== Test 3: reconstruction n=%d ===\n", n);
    double *orig = malloc((size_t)n * n * sizeof(double));
    double *fact = malloc((size_t)n * n * sizeof(double));
    if (!orig || !fact) { printf("  SKIP (malloc)\n"); free(orig); free(fact); return; }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) orig[i*n+j] = corr(i, j, n);
        orig[i*n+i] = 1.0;
    }
    memcpy(fact, orig, (size_t)n * n * sizeof(double));
    mphil_dis_cholesky(fact, n);

    double max_err = 0.0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            int kmax = i < j ? i : j;
            for (int k = 0; k <= kmax; k++) sum += fact[i*n+k] * fact[j*n+k];
            double e = fabs(sum - orig[i*n+j]);
            if (e > max_err) max_err = e;
        }

    char label[64];
    snprintf(label, sizeof(label), "max|L*L^T - C| < 1e-10  (got %.2e)", max_err);
    check(label, max_err < 1e-10);
    free(orig); free(fact);
}

/* --- Test 4: bounds check --- */
static void test_bounds(void)
{
    printf("\n=== Test 4: bounds check ===\n");
    double c[1] = {4.0};
    check("n=0 returns -1",      mphil_dis_cholesky(c, 0)      == -1.0);
    check("n=100001 returns -1", mphil_dis_cholesky(c, 100001) == -1.0);
    check("n=1 returns >= 0",    mphil_dis_cholesky(c, 1)      >= 0.0);
    check("n=1 c[0]=sqrt(4)=2",  fabs(c[0] - 2.0) < TOL);
}

int main(void)
{
    printf("=== mphil_dis_cholesky correctness tests ===\n");
    test_2x2();
    test_3x3();
    test_reconstruction(5);
    test_reconstruction(50);
    test_reconstruction(200);
    test_bounds();

    printf("\n--- %d/%d tests passed ---\n", tests_run - tests_failed, tests_run);
    return tests_failed == 0 ? 0 : 1;
}
