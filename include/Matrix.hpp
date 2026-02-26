// Matrix.hpp
// Purpose:
//   Define a minimal Matrix type (likely row-major contiguous storage) to support LU factorization.
// Suggested contents (no requirement):
//   - dimensions (n, m)
//   - std::vector<double> storage
//   - operator()(i,j) for indexing
//   - helpers like swap_rows(i,j)
//
// Keep this thin: it's an indexing wrapper, not a full linear algebra library.
