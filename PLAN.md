# Coursework Implementation Plan

## Deadline
23:59 Wednesday 1 April 2026

## Suggested Milestone Order

### Stage 1 — Baseline (tag: v0.1-baseline)
**Goal**: Get a correct, single-threaded implementation working.

1. Create `include/mphil_dis_cholesky.h` with the function signature
2. Create `src/mphil_dis_cholesky.c` with the exact baseline loop from the spec
3. Add wall-clock timing (`clock_gettime(CLOCK_MONOTONIC)`) wrapping only the computation
4. Add the n <= 100000 bounds check
5. Create `example/example.c` using the provided `corr()` matrix generator; verify the
   2×2 example from the spec (C2 → L2) prints correctly
6. Write a minimal `Makefile`
7. **Tag commit: `v0.1-baseline`**

### Stage 2 — Serial Optimisations (tag: v0.2-serial-opt)
**Goal**: Improve single-thread performance before adding parallelism.

Key ideas (apply incrementally, benchmark each):
- **Loop reorder** (highest impact): swap inner loops in the submatrix update so the
  innermost loop iterates over j (stride-1 in row-major), not i (stride-n).
  Before: `for j { for i { C[i,j] -= C[i,p]*C[p,j] } }`  (column access — cache miss)
  After:  `for i { for j { C[i,j] -= C[i,p]*C[p,j] } }`  (row access — cache-friendly)
- **Hoist loop-invariant**: precompute `c_ip = c[i*n+p]` outside the j loop.
- **Replace division with multiply**: compute `inv_diag = 1.0/diag` once; multiply instead
  of dividing in the j and i update loops.
- **Compiler flags**: use `-O3 -march=native -ffast-math` and measure the difference.
- Benchmark at n = 1000, 2000, 4000 on local machine and on CSD3.
7. **Tag commit: `v0.2-serial-opt`**

### Stage 3 — OpenMP Basic Parallelisation (tag: v0.3-openmp-v1)
**Goal**: First working, commented parallel version.

Strategy:
- The outer p-loop is SEQUENTIAL (each step depends on previous).
- Create ONE persistent `#pragma omp parallel` region OUTSIDE the p-loop.
  This avoids repeated thread creation/destruction overhead.
- Inside the p-loop:
  - `#pragma omp single` for diagonal update (one thread, implicit barrier after)
  - `#pragma omp for schedule(static)` for row update (parallel, ~O(n) work)
  - `#pragma omp for schedule(static)` for column update (parallel, ~O(n) work)
  - `#pragma omp for schedule(static)` for submatrix update outer-i loop (parallel, O(n²) work)
    — this is the dominant kernel; each row i is independent → no race conditions
- Every OpenMP pragma must have an inline comment: what it does + why.
- Store `diag` as a `shared` variable declared before the parallel region.
- Benchmark: vary OMP_NUM_THREADS (1,2,4,8,16) at n=4000,8000.
- **Tag commit: `v0.3-openmp-v1`**

### Stage 4 — Cache-Blocked (Tiled) Cholesky (tag: v0.4-blocked)
**Goal**: Improve cache reuse for large matrices.

Strategy:
- Divide the matrix into B×B tiles (B ≈ 64–128, tune experimentally).
- Blocked Cholesky steps per panel p (each of width B):
  1. Factor the B×B diagonal tile (call serial/OpenMP factoriser recursively)
  2. Solve L for all tiles in the current block-column below diagonal (TRSM, parallelisable)
  3. Update the trailing submatrix with rank-B update (DGEMM-like, parallelisable with OpenMP)
- Use `#pragma omp parallel for` over block-rows in steps 2 and 3.
- OpenMP tasks are an alternative but introduce synchronisation complexity.
- Benchmark varying B and thread counts; choose optimal B.
- **Tag commit: `v0.4-blocked`**

### Stage 5 — Final Tuning (tag: v0.5-tuned)
**Goal**: Squeeze out remaining performance.

Ideas:
- Experiment with `schedule(dynamic, chunk)` vs `schedule(static)` — static is usually
  better here because work per iteration decreases monotonically (could try `dynamic`)
- `OMP_PROC_BIND=close`, `OMP_PLACES=cores` for thread affinity
- Prefetch hints if beneficial
- Loop unrolling (let compiler do it via `-funroll-loops`)
- Run full scaling study on CSD3 icelake (1,2,4,8,16,32,48,64 cores)
- **Tag commit: `v0.5-tuned`**

### Stage 6 — Tests, Scripts, Report
- Complete `test/test_correctness.c`: compare log|C| against LAPACK/reference for
  several matrix sizes; print PASS/FAIL
- Complete `test/test_performance.c`: automated timing sweep over n and thread counts
- Complete `scripts/benchmark.sh`: batch runner for CSD3 timing experiments
- Complete `example/submit_csd3.slurm`: SLURM script for icelake partition
- Write `report/report.pdf` (< 3000 words):
  - Tagged commits table
  - Parallelisation strategy description
  - Performance plots: time vs n (log-log), speedup vs threads
  - Tables with compilation flags, platform, thread counts
  - Discussion of bottlenecks and what helped most

## Performance Metric
GFlop/s = (n³/3) / time / 1e9
(Cholesky is ~n³/3 floating point operations)

## CSD3 Usage
- Partition: `icelake` (76 cores/node, best for OpenMP)
- Compile on login node: `gcc -O3 -march=native -fopenmp -ffast-math`
- Request whole node for clean benchmarks: `--exclusive`

## Report Word Budget (< 3000 words)
- Introduction/background: ~200 words
- Implementation & optimisation strategy: ~800 words
- Results (with figures/tables): ~1200 words
- Tagged commits table: ~100 words
- Conclusion: ~200 words
- Total: ~2500 words (comfortable margin)
