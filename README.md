# C2 HPC Coursework — Cholesky Factorization with OpenMP

MPhil in Data Intensive Science, Lent Term 2026

## Overview

This project implements a high-performance, OpenMP-parallelised Cholesky factorisation
routine (`C = L L^T`) for symmetric positive-definite matrices in C.

## Directory Structure

```
src/         Library source code (mphil_dis_cholesky.c)
include/     Public header (mphil_dis_cholesky.h)
example/     Example program + CSD3 SLURM submission scripts
test/        Correctness and performance test programs
scripts/     Benchmarking and plotting helper scripts
report/      PDF report (report.pdf — final submission)
Makefile     Build system
README.md    This file
```

## Function Interface

```c
#include "mphil_dis_cholesky.h"

double mphil_dis_cholesky(double *c, int n);
```

- **Input**: `c` — pointer to `n*n` doubles in row-major order (symmetric positive-definite matrix C)
- **Input**: `n` — matrix dimension (must satisfy `n <= 100000`)
- **In-place**: overwrites lower triangle and diagonal with L; upper triangle with L^T
- **Returns**: wall clock time (seconds) elapsed during factorisation

## Building the Library

### Prerequisites

- GCC with OpenMP support (gcc >= 9)
- On CSD3: `module load gcc/11`

### Compile

```bash
make          # builds shared library lib/libcholesky.so and static lib/libcholesky.a
make example  # builds example/example
make test     # builds and runs tests
make clean    # removes build artefacts
```

### Recommended compilation flags (used by Makefile)

```
-O3 -march=native -fopenmp -ffast-math
```

## Performance Guidance

- Set the number of OpenMP threads via the environment variable before running:
  ```bash
  export OMP_NUM_THREADS=16
  ```
- For best performance on CSD3 icelake nodes, match `OMP_NUM_THREADS` to the number
  of physical cores allocated (up to 76 per node).
- Pin threads to cores to avoid NUMA effects:
  ```bash
  export OMP_PROC_BIND=close
  export OMP_PLACES=cores
  ```
- The routine scales well for `n >= 2000`; for small matrices the serial overhead dominates.

## Running on CSD3

See `example/submit_csd3.slurm` for a ready-to-use SLURM submission script.

```bash
sbatch example/submit_csd3.slurm
```

## Correctness Verification

Tests in `test/` compare `log|C|` computed from the returned L matrix against a
reference value from LAPACK `dpotrf`. Run with:

```bash
make test
./test/test_correctness
```

## Git Tag Index

| Tag            | Description                                      |
|----------------|--------------------------------------------------|
| v0.1-baseline  | Single-threaded naive implementation             |
| v0.2-serial-opt| Serial optimisations: loop reorder, precompute   |
| v0.3-openmp-v1 | First working OpenMP parallel version            |
| v0.4-blocked   | Blocked/tiled Cholesky for cache efficiency      |
| v0.5-tuned     | Final tuned version with scheduling experiments  |
