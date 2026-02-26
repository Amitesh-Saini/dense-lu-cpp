// test_lu.cpp
// Purpose:
//   Unit tests for LU factorization + solve.
//
// What to test:
//   - Factorization correctness via residual:
//       ||P*A - L*U|| / ||A||
//   - Solve correctness:
//       ||A*x - b|| / ||b||
//   - Edge cases:
//       * singular matrices should fail factorization
//       * small sizes (2x2, 3x3) with known answers
//       * random matrices across several sizes
//
// Wire this into CTest via CMake.
