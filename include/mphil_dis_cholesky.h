#ifndef MPHIL_DIS_CHOLESKY_H
#define MPHIL_DIS_CHOLESKY_H

/* Computes Cholesky factorization C = L L^T in-place.
 * c: n*n matrix in row-major order, overwritten with L (lower) and L^T (upper).
 * Returns wall clock time in seconds, or -1.0 if n is out of range [1, 100000]. */
double mphil_dis_cholesky(double *c, int n);

#endif
