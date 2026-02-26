// bench_lu.cpp
// Purpose:
//   Benchmark LU factorization (and later solve) across sizes.
//
// Suggested behavior:
//   - Generate random matrices for n in {16, 32, 64, 128, ...}
//   - Time factorization with std::chrono
//   - Write results to data/bench.csv (size, time_ms, residual, etc.)
//
// Keep benchmarks separate from correctness tests.
