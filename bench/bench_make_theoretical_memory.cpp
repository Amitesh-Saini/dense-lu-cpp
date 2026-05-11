#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::size_t double_bytes = sizeof(double);
constexpr std::size_t size_t_bytes = sizeof(std::size_t);

double bytes_to_mb(std::size_t bytes)
{
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

void write_single_rhs_memory_csv(const std::filesystem::path& output_path, const std::vector<std::size_t>& sizes){

    
    std::filesystem::create_directories(output_path.parent_path());

    std::ofstream out(output_path);

    if (!out.is_open()) {
        throw std::runtime_error("Failed to open output file: " + output_path.string());
    }

    out << "n,"
        << "A0_bytes,"
        << "LU_bytes,"
        << "b_bytes,"
        << "x_true_bytes,"
        << "x_computed_bytes,"
        << "pivot_bytes,"
        << "total_bytes,"
        << "total_mb\n";

    for (const std::size_t n : sizes) {
        const std::size_t A0_bytes = n * n * double_bytes;
        const std::size_t LU_bytes = n * n * double_bytes;

        const std::size_t b_bytes = n * double_bytes;
        const std::size_t x_true_bytes = n * double_bytes;
        const std::size_t x_computed_bytes = n * double_bytes;

        const std::size_t pivot_bytes = n * size_t_bytes;

        const std::size_t total_bytes =
            A0_bytes
            + LU_bytes
            + b_bytes
            + x_true_bytes
            + x_computed_bytes
            + pivot_bytes;

        out << n << ','
            << A0_bytes << ','
            << LU_bytes << ','
            << b_bytes << ','
            << x_true_bytes << ','
            << x_computed_bytes << ','
            << pivot_bytes << ','
            << total_bytes << ','
            << bytes_to_mb(total_bytes) << '\n';
    }
}

void write_multiple_rhs_memory_csv(const std::filesystem::path& output_path,
                                   const std::vector<std::size_t>& sizes,
                                   const std::vector<std::size_t>& rhs_counts)
{
    std::filesystem::create_directories(output_path.parent_path());

    std::ofstream out(output_path);

    if (!out.is_open()) {
        throw std::runtime_error("Failed to open output file: " + output_path.string());
    }

    out << "n,"
        << "nrhs,"
        << "A0_bytes,"
        << "LU_bytes,"
        << "X_true_bytes,"
        << "B_bytes,"
        << "X_computed_bytes,"
        << "pivot_bytes,"
        << "temporary_rhs_work_bytes,"
        << "total_bytes,"
        << "total_mb\n";

    for (const std::size_t n : sizes) {
        for (const std::size_t nrhs : rhs_counts) {
            const std::size_t A0_bytes = n * n * double_bytes;
            const std::size_t LU_bytes = n * n * double_bytes;

            const std::size_t X_true_bytes = n * nrhs * double_bytes;
            const std::size_t B_bytes = n * nrhs * double_bytes;
            const std::size_t X_computed_bytes = n * nrhs * double_bytes;

            const std::size_t pivot_bytes = n * size_t_bytes;

            // Temporary vectors used during column-wise multiple-RHS solving:
            // x_i_computed and b_i.
            const std::size_t temporary_rhs_work_bytes = 2 * n * double_bytes;

            const std::size_t total_bytes =
                A0_bytes
                + LU_bytes
                + X_true_bytes
                + B_bytes
                + X_computed_bytes
                + pivot_bytes
                + temporary_rhs_work_bytes;

            out << n << ','
                << nrhs << ','
                << A0_bytes << ','
                << LU_bytes << ','
                << X_true_bytes << ','
                << B_bytes << ','
                << X_computed_bytes << ','
                << pivot_bytes << ','
                << temporary_rhs_work_bytes << ','
                << total_bytes << ','
                << bytes_to_mb(total_bytes) << '\n';
        }
    }
}

} // namespace

int main()
{
    try {
        const std::vector<std::size_t> single_rhs_sizes = {
            8, 16, 32, 64, 128, 256, 512
        };

        const std::vector<std::size_t> multiple_rhs_sizes = {
            8, 16, 32, 64, 128, 256
        };

        const std::vector<std::size_t> rhs_counts = {
            1, 2, 4, 8, 16
        };

        const std::filesystem::path output_dir =
            "data/benchmarks_theoretical_memory";

        write_single_rhs_memory_csv(
            output_dir / "theoretical_memory_single_rhs.csv",
            single_rhs_sizes
        );

        write_multiple_rhs_memory_csv(
            output_dir / "theoretical_memory_multiple_rhs.csv",
            multiple_rhs_sizes,
            rhs_counts
        );

        std::cout << "Wrote theoretical memory CSVs to "
                  << output_dir.string() << "/\n";

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Failed to write theoretical memory CSVs: "
                  << e.what() << '\n';
        return 1;
    }
}