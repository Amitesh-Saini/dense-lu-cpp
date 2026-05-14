# plot_bench.py
#
# Purpose:
#   Generate publication-quality plots from LU benchmark CSV files.
#   Applies a consistent HPC/scientific style: LaTeX-style fonts, tight axis
#   formatting, reference complexity lines, and IEEEtran-compatible colour palette.
#
# Usage:
#   python plot_bench.py
#   All figures are written to plots/ at 300 dpi.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR  = Path("data")
PLOTS_DIR = Path("plots")


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

# Colour-blind-safe, print-friendly palette (Paul Tol "bright")
COLOURS = {
    "custom_lu":              "#4477AA",   # blue
    "eigen_partial_piv_lu":  "#EE6677",   # red
    "factorization_residual": "#4477AA",
    "solve_residual":         "#CCBB44",   # yellow
    "solution_error_inf":     "#228833",   # green
    "residual_factorization_debug": "#4477AA",
    "residual_factorization_fast":  "#EE6677",
    "residual_solve":               "#228833",
    "single_rhs":    "#4477AA",
    "multiple_rhs":  "#EE6677",
}

MARKERS = {
    "custom_lu":             "o",
    "eigen_partial_piv_lu":  "s",
}

IMPLEMENTATION_LABELS = {
    "custom_lu":             "Custom LU",
    "eigen_partial_piv_lu":  "Eigen PartialPivLU",
}

PHASE_LABELS = {
    "factorization":                   "Factorization",
    "solve":                           "Solve",
    "full_solve":                      "Full solve",
    "residual_factorization_debug":    r"Fact. residual (debug, $\|PA - LU\|_\infty$)",
    "residual_factorization_fast":     r"Fact. residual (fast, $\|PA - LU\|_\infty$)",
    "residual_solve":                  r"Solve residual ($\|Ax - b\|_\infty$)",
    "multiple_rhs":                    "Multiple RHS",
    "multiple_rhs_residual":           "Multiple RHS residual",
}

MATRIX_TYPE_LABELS = {
    "random_dense":        "Random dense",
    "diagonally_dominant": "Diag. dominant",
    "hilbert":             "Hilbert",
    "pivot_stress":        "Pivot stress",
    "near_singular":       "Near singular",
    "singular":            "Singular",
}

# Shared rcParams applied once at module load
_RC = {
    # Font — use TeX-style sizing without requiring a full LaTeX install
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "legend.framealpha": 0.85,
    "legend.edgecolor":  "0.7",
    # Lines
    "lines.linewidth":  1.6,
    "lines.markersize": 5,
    # Grid
    "axes.grid":        True,
    "grid.linestyle":   "--",
    "grid.linewidth":   0.4,
    "grid.color":       "0.75",
    "axes.axisbelow":   True,
    # Spines
    "axes.spines.top":   False,
    "axes.spines.right": False,
    # Figure
    "figure.figsize":   (6.0, 4.2),   # single-column IEEE width
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
}

mpl.rcParams.update(_RC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log2_x_ticks(ax: plt.Axes, values: list[int]) -> None:
    """Set log2 x-axis with plain integer tick labels."""
    ax.set_xscale("log", base=2)
    ax.set_xticks(values)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())


def _add_complexity_line(
    ax: plt.Axes,
    xs: np.ndarray,
    ys: np.ndarray,
    exponent: int,
    label: str,
    *,
    colour: str = "0.45",
) -> None:
    """
    Overlay a reference O(n^exponent) trend line anchored to the last data point.

    Anchoring to the last point keeps the line in frame without any manual scaling.
    """
    x_ref, y_ref = xs[-1], ys[-1]
    # Normalise so the line passes through (x_ref, y_ref)
    y_ref_theory = x_ref ** exponent
    scale = y_ref / y_ref_theory
    y_theory = scale * xs ** exponent
    ax.plot(
        xs, y_theory,
        color=colour, linestyle=":", linewidth=1.0,
        label=label, zorder=1,
    )


def _save(ax: plt.Axes, filename: str) -> None:
    """Apply final polish and write figure to PLOTS_DIR."""
    fig = ax.get_figure()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / filename
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csvs(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    """
    Load all benchmark CSV files produced by the C++ benchmark suite.

    Reads timing, accuracy, residual timing, measured memory, multiple-RHS,
    and theoretical memory CSVs from data/.  Raises FileNotFoundError on any
    missing file so the caller gets an actionable message immediately.
    """
    csv_paths: dict[str, Path] = {
        "main_timing":          data_dir / "benchmarks_main"              / "timing.csv",
        "main_accuracy":        data_dir / "benchmarks_main"              / "accuracy.csv",
        "main_residual_timing": data_dir / "benchmarks_main"              / "residual_timing.csv",
        "main_memory":          data_dir / "benchmarks_main"              / "memory.csv",

        "stress_timing":          data_dir / "benchmarks_stress"          / "timing.csv",
        "stress_accuracy":        data_dir / "benchmarks_stress"          / "accuracy.csv",
        "stress_residual_timing": data_dir / "benchmarks_stress"          / "residual_timing.csv",
        "stress_memory":          data_dir / "benchmarks_stress"          / "memory.csv",

        "hilbert_timing":          data_dir / "benchmarks_hilbert"        / "timing.csv",
        "hilbert_accuracy":        data_dir / "benchmarks_hilbert"        / "accuracy.csv",
        "hilbert_residual_timing": data_dir / "benchmarks_hilbert"        / "residual_timing.csv",
        "hilbert_memory":          data_dir / "benchmarks_hilbert"        / "memory.csv",

        "singular_timing":   data_dir / "benchmarks_singular"             / "timing.csv",
        "singular_accuracy": data_dir / "benchmarks_singular"             / "accuracy.csv",
        "singular_memory":   data_dir / "benchmarks_singular"             / "memory.csv",

        "multiple_rhs":                 data_dir / "benchmarks_multiple_rhs" / "multiple_rhs.csv",
        "multiple_rhs_residual_timing": data_dir / "benchmarks_multiple_rhs" / "multiple_rhs_residual_timing.csv",
        "multiple_rhs_memory":          data_dir / "benchmarks_multiple_rhs" / "memory.csv",

        "theoretical_memory_single_rhs":   data_dir / "benchmarks_theoretical_memory" / "theoretical_memory_single_rhs.csv",
        "theoretical_memory_multiple_rhs": data_dir / "benchmarks_theoretical_memory" / "theoretical_memory_multiple_rhs.csv",

        # Large-n suite (n = 1024, 2048 — single trial, no residual timing)
        "large_n_timing": data_dir / "benchmarks_large_n" / "timing.csv",
    }

    data: dict[str, pd.DataFrame] = {}
    for name, path in csv_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing CSV '{name}': {path}")
        data[name] = pd.read_csv(path)

    return data


def median_by_group(
    df: pd.DataFrame,
    group_cols: list[str],
    value_cols: list[str],
) -> pd.DataFrame:
    """
    Reduce repeated trials to per-group medians.

    Groups by group_cols (implementation, matrix_type, phase, n, …) and
    returns the column-wise median of value_cols, sorted by group_cols.
    """
    missing_g = [c for c in group_cols if c not in df.columns]
    missing_v = [c for c in value_cols if c not in df.columns]
    if missing_g:
        raise KeyError(f"Missing group columns: {missing_g}")
    if missing_v:
        raise KeyError(f"Missing value columns: {missing_v}")

    return (
        df.groupby(group_cols, as_index=False)[value_cols]
        .median()
        .sort_values(group_cols)
        .reset_index(drop=True)
    )


def filter_success(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows where status == 'success'; pass through if no status column."""
    if "status" not in df.columns:
        return df.copy()
    return df[df["status"] == "success"].copy()


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_factorization_time_vs_n(data: dict[str, pd.DataFrame]) -> None:
    """
    Custom LU vs Eigen factorization time — expected O(n^3) scaling.

    Both curves are drawn; an O(n^3) reference line is anchored to the Custom
    LU series so deviations from cubic scaling are immediately visible.
    """
    df = filter_success(data["main_timing"])
    df = df[(df["matrix_type"] == "random_dense") & (df["phase"] == "factorization")]

    summary = median_by_group(
        df,
        group_cols=["implementation", "matrix_type", "phase", "n"],
        value_cols=["time_ms"],
    )

    fig, ax = plt.subplots()

    for impl, group in summary.groupby("implementation"):
        group = group.sort_values("n")
        ax.plot(
            group["n"], group["time_ms"],
            marker=MARKERS.get(impl, "o"),
            color=COLOURS.get(impl),
            label=IMPLEMENTATION_LABELS.get(impl, impl),
        )

    # Reference complexity line anchored to Custom LU
    custom = summary[summary["implementation"] == "custom_lu"].sort_values("n")
    if not custom.empty:
        _add_complexity_line(
            ax,
            custom["n"].to_numpy(float),
            custom["time_ms"].to_numpy(float),
            exponent=3,
            label=r"$O(n^3)$ reference",
        )

    ns = sorted(summary["n"].unique())
    _log2_x_ticks(ax, ns)
    ax.set_yscale("log")
    ax.set_xlabel("Matrix size $n$")
    ax.set_ylabel("Median factorization time (ms)")
    ax.set_title("LU Factorization Time vs. Matrix Size")
    ax.legend()

    _save(ax, "factorization_time_vs_n.png")


def plot_solve_time_vs_n(data: dict[str, pd.DataFrame]) -> None:
    """
    Custom LU vs Eigen triangular solve time — expected O(n^2) scaling.
    """
    df = filter_success(data["main_timing"])
    df = df[(df["matrix_type"] == "random_dense") & (df["phase"] == "solve")]

    summary = median_by_group(
        df,
        group_cols=["implementation", "matrix_type", "phase", "n"],
        value_cols=["time_ms"],
    )

    fig, ax = plt.subplots()

    for impl, group in summary.groupby("implementation"):
        group = group.sort_values("n")
        ax.plot(
            group["n"], group["time_ms"],
            marker=MARKERS.get(impl, "o"),
            color=COLOURS.get(impl),
            label=IMPLEMENTATION_LABELS.get(impl, impl),
        )

    custom = summary[summary["implementation"] == "custom_lu"].sort_values("n")
    if not custom.empty:
        _add_complexity_line(
            ax,
            custom["n"].to_numpy(float),
            custom["time_ms"].to_numpy(float),
            exponent=2,
            label=r"$O(n^2)$ reference",
        )

    ns = sorted(summary["n"].unique())
    _log2_x_ticks(ax, ns)
    ax.set_yscale("log")
    ax.set_xlabel("Matrix size $n$")
    ax.set_ylabel("Median solve time (ms)")
    ax.set_title("Triangular Solve Time vs. Matrix Size")
    ax.legend()

    _save(ax, "solve_time_vs_n.png")


def plot_accuracy_residuals_vs_n(data: dict[str, pd.DataFrame]) -> None:
    """
    Factorization residual, solve residual, and forward error for Custom LU.

    All three metrics are in the same log-y plot so relative magnitudes are
    directly comparable.  Dashed machine-epsilon band at 2^-52 is included for
    reference.
    """
    df = filter_success(data["main_accuracy"])
    df = df[
        (df["implementation"] == "custom_lu")
        & (df["matrix_type"] == "random_dense")
    ]

    summary = median_by_group(
        df,
        group_cols=["implementation", "matrix_type", "n"],
        value_cols=["factorization_residual", "solve_residual", "solution_error_inf"],
    ).sort_values("n")

    fig, ax = plt.subplots()

    series = [
        ("factorization_residual", "Factorization residual"),
        ("solve_residual",         r"Solve residual $\|Ax-b\|_\infty / \|b\|_\infty$"),
        ("solution_error_inf",     r"Forward error $\|x - x^*\|_\infty$"),
    ]

    for col, label in series:
        ax.plot(
            summary["n"], summary[col],
            marker="o",
            color=COLOURS.get(col),
            label=label,
        )

    # Machine epsilon reference
    eps = np.finfo(float).eps
    ns = sorted(summary["n"].unique())
    ax.axhline(eps, color="0.5", linestyle=":", linewidth=0.9, label=r"$\epsilon_{\mathrm{mach}}$")

    _log2_x_ticks(ax, ns)
    ax.set_yscale("log")
    ax.set_xlabel("Matrix size $n$")
    ax.set_ylabel("Median relative value")
    ax.set_title("Custom LU — Accuracy and Residuals")
    ax.legend(fontsize=8)

    _save(ax, "accuracy_residuals_vs_n.png")


def plot_hilbert_solution_error(data: dict[str, pd.DataFrame]) -> None:
    """
    Forward error on Hilbert matrices — demonstrates condition-number growth.

    A unit-error reference line (value = 1) is drawn to mark where the
    solution is completely wrong; both solvers lose accuracy at the same rate
    because the ill-conditioning is intrinsic to the Hilbert matrix, not to
    the algorithm.
    """
    df = filter_success(data["hilbert_accuracy"])
    df = df[df["solution_error_inf"].notna() & (df["solution_error_inf"] > 0.0)]

    summary = median_by_group(
        df,
        group_cols=["implementation", "matrix_type", "n"],
        value_cols=["solution_error_inf"],
    )

    fig, ax = plt.subplots()

    for impl, group in summary.groupby("implementation"):
        group = group.sort_values("n")
        ax.plot(
            group["n"], group["solution_error_inf"],
            marker=MARKERS.get(impl, "o"),
            color=COLOURS.get(impl),
            label=IMPLEMENTATION_LABELS.get(impl, impl),
        )

    ns = sorted(summary["n"].unique())
    ax.axhline(1.0, color="0.5", linestyle=":", linewidth=0.9, label="Unit error")
    ax.set_xticks(ns)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_yscale("log")
    ax.set_xlabel("Matrix size $n$")
    ax.set_ylabel(r"Median forward error $\|x - x^*\|_\infty$")
    ax.set_title("Hilbert Matrix — Forward Error vs. $n$")
    ax.legend()

    _save(ax, "hilbert_solution_error.png")


def plot_multiple_rhs_total_time(data: dict[str, pd.DataFrame]) -> None:
    """
    Total solve time vs. number of RHS vectors at fixed n = 256.

    Custom LU cost grows as O(n_rhs) because each RHS requires a fresh
    triangular solve; Eigen's batched solve can amortise some overhead.
    """
    df = filter_success(data["multiple_rhs"])
    df = df[(df["matrix_type"] == "random_dense") & (df["n"] == 256)]

    summary = median_by_group(
        df,
        group_cols=["implementation", "matrix_type", "n", "nrhs"],
        value_cols=["total_time_ms"],
    )

    fig, ax = plt.subplots()

    for impl, group in summary.groupby("implementation"):
        group = group.sort_values("nrhs")
        ax.plot(
            group["nrhs"], group["total_time_ms"],
            marker=MARKERS.get(impl, "o"),
            color=COLOURS.get(impl),
            label=IMPLEMENTATION_LABELS.get(impl, impl),
        )

    nrhs_vals = sorted(summary["nrhs"].unique())
    ax.set_xscale("log", base=2)
    ax.set_xticks(nrhs_vals)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlabel("Number of RHS vectors")
    ax.set_ylabel("Median total time (ms)")
    ax.set_title("Multiple-RHS Total Time  ($n = 256$)")
    ax.legend()

    _save(ax, "multiple_rhs_total_time.png")


def plot_multiple_rhs_time_per_rhs(data: dict[str, pd.DataFrame]) -> None:
    """
    Time per RHS vs. number of RHS vectors at fixed n = 256.

    For Custom LU this is flat (each solve is independent); Eigen's batched
    kernel amortises overhead so time per RHS decreases.
    """
    df = filter_success(data["multiple_rhs"])
    df = df[(df["matrix_type"] == "random_dense") & (df["n"] == 256)]

    summary = median_by_group(
        df,
        group_cols=["implementation", "matrix_type", "n", "nrhs"],
        value_cols=["time_per_rhs_ms"],
    )

    fig, ax = plt.subplots()

    for impl, group in summary.groupby("implementation"):
        group = group.sort_values("nrhs")
        ax.plot(
            group["nrhs"], group["time_per_rhs_ms"],
            marker=MARKERS.get(impl, "o"),
            color=COLOURS.get(impl),
            label=IMPLEMENTATION_LABELS.get(impl, impl),
        )

    nrhs_vals = sorted(summary["nrhs"].unique())
    ax.set_xscale("log", base=2)
    ax.set_xticks(nrhs_vals)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlabel("Number of RHS vectors")
    ax.set_ylabel("Median time per RHS (ms)")
    ax.set_title("Multiple-RHS Time per RHS  ($n = 256$)")
    ax.legend()

    _save(ax, "multiple_rhs_time_per_rhs.png")


def plot_residual_timing_vs_n(data: dict[str, pd.DataFrame]) -> None:
    """
    Residual-check cost vs. matrix size.

    Three series: full PA-LU residual (debug), fast factorization residual,
    and solve residual.  The debug PA-LU check is O(n^3) (matrix multiply),
    so it quickly dominates solver time at large n.
    """
    df = data["main_residual_timing"]

    phases = [
        "residual_factorization_debug",
        "residual_factorization_fast",
        "residual_solve",
    ]

    df = df[
        (df["matrix_type"] == "random_dense")
        & (df["phase"].isin(phases))
    ]

    summary = median_by_group(
        df,
        group_cols=["matrix_type", "phase", "n"],
        value_cols=["time_ms"],
    )

    fig, ax = plt.subplots()

    for phase, group in summary.groupby("phase"):
        group = group.sort_values("n")
        ax.plot(
            group["n"], group["time_ms"],
            marker="o",
            color=COLOURS.get(phase),
            label=PHASE_LABELS.get(phase, phase),
        )

    ns = sorted(summary["n"].unique())
    _log2_x_ticks(ax, ns)
    ax.set_yscale("log")
    ax.set_xlabel("Matrix size $n$")
    ax.set_ylabel("Median residual-check time (ms)")
    ax.set_title("Residual Verification Cost vs. Matrix Size")
    ax.legend(fontsize=8)

    _save(ax, "residual_timing_vs_n.png")


def plot_theoretical_memory_vs_n(data: dict[str, pd.DataFrame]) -> None:
    """
    Theoretical memory footprint vs. matrix size.

    Both single-RHS and 16-RHS cases are shown to illustrate that the n x n
    LU factor dominates; the extra RHS storage is negligible until n is large.
    An O(n^2) reference confirms the expected storage complexity.
    """
    single   = data["theoretical_memory_single_rhs"].copy()
    multiple = data["theoretical_memory_multiple_rhs"].copy()

    multiple = multiple[multiple["nrhs"] == 16].copy()
    shared_n = sorted(set(single["n"]).intersection(set(multiple["n"])))

    single   = single[single["n"].isin(shared_n)].sort_values("n")
    multiple = multiple[multiple["n"].isin(shared_n)].sort_values("n")

    fig, ax = plt.subplots()

    ax.plot(
        single["n"], single["total_mb"],
        marker="o",
        color=COLOURS["single_rhs"],
        label="Single RHS",
    )
    ax.plot(
        multiple["n"], multiple["total_mb"],
        marker="s",
        color=COLOURS["multiple_rhs"],
        label="Multiple RHS ($n_\\mathrm{rhs} = 16$)",
    )

    _add_complexity_line(
        ax,
        single["n"].to_numpy(float),
        single["total_mb"].to_numpy(float),
        exponent=2,
        label=r"$O(n^2)$ reference",
    )

    _log2_x_ticks(ax, shared_n)
    ax.set_yscale("log")
    ax.set_xlabel("Matrix size $n$")
    ax.set_ylabel("Theoretical memory (MB)")
    ax.set_title("Theoretical Memory Footprint vs. Matrix Size")
    ax.legend()

    _save(ax, "theoretical_memory_vs_n.png")


def plot_multiple_rhs_solve_loop_time(data: dict[str, pd.DataFrame]) -> None:
    """
    Solve-loop time (factorization excluded) vs. number of RHS vectors at n = 256.

    Isolates the pure triangular-solve cost to separate it from factorization
    reuse overhead.
    """
    df = filter_success(data["multiple_rhs"])
    df = df[(df["matrix_type"] == "random_dense") & (df["n"] == 256)]

    summary = median_by_group(
        df,
        group_cols=["implementation", "matrix_type", "n", "nrhs"],
        value_cols=["solve_loop_time_ms"],
    )

    fig, ax = plt.subplots()

    for impl, group in summary.groupby("implementation"):
        group = group.sort_values("nrhs")
        ax.plot(
            group["nrhs"], group["solve_loop_time_ms"],
            marker=MARKERS.get(impl, "o"),
            color=COLOURS.get(impl),
            label=IMPLEMENTATION_LABELS.get(impl, impl),
        )

    nrhs_vals = sorted(summary["nrhs"].unique())
    ax.set_xscale("log", base=2)
    ax.set_xticks(nrhs_vals)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlabel("Number of RHS vectors")
    ax.set_ylabel("Median solve-loop time (ms)")
    ax.set_title("Multiple-RHS Solve Time  ($n = 256$, factorization excluded)")
    ax.legend()

    _save(ax, "multiple_rhs_solve_loop_time.png")


def plot_large_n_factorization_comparison(data: dict[str, pd.DataFrame]) -> None:
    """
    Full-range factorization comparison: Custom LU vs Eigen, n = 8 … 2048.

    Merges the main benchmark suite (n = 8 … 512) with the large-n suite
    (n = 1024, 2048) so the reader sees the complete scaling curve in one
    figure.  The O(n³) reference line is anchored to the midpoint of the
    Custom LU series so it sits visually between the two data curves rather
    than collapsing onto either endpoint.
    """
    # Merge main and large-n timing; keep only random_dense factorization rows.
    combined = pd.concat(
        [data["main_timing"], data["large_n_timing"]],
        ignore_index=True,
    )
    combined = filter_success(combined)
    combined = combined[
        (combined["matrix_type"] == "random_dense")
        & (combined["phase"] == "factorization")
    ]

    summary = median_by_group(
        combined,
        group_cols=["implementation", "matrix_type", "phase", "n"],
        value_cols=["time_ms"],
    )

    fig, ax = plt.subplots()

    for impl, group in summary.groupby("implementation"):
        group = group.sort_values("n")
        ax.plot(
            group["n"], group["time_ms"],
            marker=MARKERS.get(impl, "o"),
            color=COLOURS.get(impl),
            label=IMPLEMENTATION_LABELS.get(impl, impl),
        )

    # Anchor the O(n^3) line to the midpoint of the Custom LU series.
    # This avoids making the reference line depend too heavily on either
    # the smallest noisy sizes or the largest one-trial sizes.
    custom = summary[summary["implementation"] == "custom_lu"].sort_values("n")
    if not custom.empty:
        ns_arr  = custom["n"].to_numpy(float)
        tms_arr = custom["time_ms"].to_numpy(float)
        mid     = len(ns_arr) // 2
        x_ref, y_ref = ns_arr[mid], tms_arr[mid]
        scale   = y_ref / x_ref ** 3
        y_ref_line = scale * ns_arr ** 3
        ax.plot(
            ns_arr, y_ref_line,
            color="0.45", linestyle=":", linewidth=1.0,
            label=r"$O(n^3)$ reference", zorder=1,
        )

    ns = sorted(summary["n"].unique())
    _log2_x_ticks(ax, ns)
    ax.set_yscale("log")
    ax.set_xlabel("Matrix size $n$")
    ax.set_ylabel("Median factorization time (ms)")
    ax.set_title(r"LU Factorization Scaling: Custom vs. Eigen ($n = 8 \ldots 2048$)")
    ax.legend()

    _save(ax, "large_n_factorization_comparison.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Load all CSVs and regenerate every benchmark figure."""
    print("Loading CSVs …")
    data = load_csvs()

    print("Generating plots …")
    plot_factorization_time_vs_n(data)
    plot_solve_time_vs_n(data)
    plot_accuracy_residuals_vs_n(data)
    plot_hilbert_solution_error(data)
    plot_multiple_rhs_total_time(data)
    plot_multiple_rhs_time_per_rhs(data)
    plot_residual_timing_vs_n(data)
    plot_theoretical_memory_vs_n(data)
    plot_multiple_rhs_solve_loop_time(data)
    plot_large_n_factorization_comparison(data)

    print("Done.")


if __name__ == "__main__":
    main()