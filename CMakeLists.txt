# CMakeLists.txt
# Purpose: CMake build configuration for the LU project.
# You will add targets such as:
#   - lu_core      (library with LU factorization + solve)
#   - lu_example   (small demo executable)
#   - lu_tests     (unit tests, wired into CTest)
#   - bench_lu     (benchmark executable that writes CSV into /data)
#
# Notes:
# - Keep flags/tooling in one place.
# - Prefer separate targets over one giant executable.
