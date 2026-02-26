// lu.cpp
// Purpose:
//   Implement LU factorization (PA = LU) and the linear solver.
//
// Implementation notes:
//   - Partial pivoting (choose max abs pivot in column).
//   - Store L and U in a single matrix (below diag = L multipliers, diag+above = U).
//   - Use a permutation vector instead of forming an explicit P matrix.
//   - Detect singular/near-singular pivots and return failure.
