/*
 * mphil_dis_cholesky.h
 *
 * Public header for the MPhil DIS Cholesky factorization library.
 *
 * Include this header in any program that uses mphil_dis_cholesky():
 *
 *   #include "mphil_dis_cholesky.h"
 *
 * Build the library first with:
 *
 *   make
 *
 * Then link your program against -lcholesky -lm.
 */

#ifndef MPHIL_DIS_CHOLESKY_H
#define MPHIL_DIS_CHOLESKY_H

/*
 * mphil_dis_cholesky
 *
 * Computes the Cholesky factorization  C = L L^T  of a symmetric
 * positive-definite n x n matrix C, in-place.
 *
 * After the call:
 *   - The lower triangle and diagonal of c[] contain L.
 *   - The upper triangle of c[] contains L^T
 *     (i.e. c[i*n+j] = L[j][i]  for i < j).
 *
 * This matches the layout required by the spec: calling the routine on
 * the 2x2 example C2 = {{4,2},{2,26}} produces {{2,1},{1,5}}.
 *
 * Parameters:
 *   c   Pointer to n*n doubles stored in row-major order.
 *       Element (i,j) is at index i*n + j.
 *       The array is modified in-place.
 *   n   Matrix dimension.  Must satisfy  1 <= n <= 100000.
 *
 * Returns:
 *   Wall clock time elapsed (in seconds) for the factorization itself.
 *   Returns -1.0 if n is outside the allowed range.
 *
 * Example:
 *   double c[4] = {4.0, 2.0, 2.0, 26.0};  // C2 from the spec
 *   double t = mphil_dis_cholesky(c, 2);
 *   // c is now {2.0, 1.0, 1.0, 5.0}
 *   // log|C| = 2*(log(c[0]) + log(c[3])) = log(100) ~ 4.6052
 */
double mphil_dis_cholesky(double *c, int n);

#endif /* MPHIL_DIS_CHOLESKY_H */
