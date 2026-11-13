"""
Batch plot OpenMOC convergence histories from all .log files in ./log/.

For every .log file, the script:
  1. Extracts iteration number, k_eff, and residual.
  2. Extracts available run/discretisation parameters:
       - number of azimuthal angles
       - azimuthal ray spacing
       - number of polar angles
       - axial (Z) spacing
       - number of threads
       - number of FSRs
       - number of 3D tracks
  3. Creates one figure with:
       - k_eff on the left y-axis (red)
       - residual on the right y-axis (blue, logarithmic scale)
  4. Saves the figure as PDF in ./plots/.
  5. Displays all generated figures after processing all log files.

Example output filename
-----------------------
convergence_a4_p2_s2p5_z5_t4_FSR99177_20261111_195508.pdf
"""

from pathlib import Path
from datetime import datetime
import re
import sys

import matplotlib.pyplot as plt


# ===========================================================================
# REGULAR EXPRESSIONS
# ===========================================================================

ITERATION_RE = re.compile(
    r"Iteration\s+(\d+):\s+"
    r"k_eff\s*=\s*([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+"
    r"res\s*=\s*([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)"
)

PARAMETER_PATTERNS = {
    "azim_angles": re.compile(
        r"Number of azimuthal angles\s*=\s*(\d+)"
    ),
    "azim_spacing": re.compile(
        r"Azimuthal ray spacing\s*=\s*([+-]?\d+(?:\.\d+)?)"
    ),
    "polar_angles": re.compile(
        r"Number of polar angles\s*=\s*(\d+)"
    ),
    "z_spacing": re.compile(
        r"Z-spacing\s*=\s*([+-]?\d+(?:\.\d+)?)"
    ),
    "threads": re.compile(
        r"Using\s+(\d+)\s+threads"
    ),
    "fsrs": re.compile(
        r"Total number of FSRs\s+(\d+)"
    ),
    "tracks": re.compile(
        r"3D Tracks in domain:\s*(\d+)"
    ),
}

DATE_RE = re.compile(
    r"Current local time and date:\s*"
    r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+"
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+"
    r"(\d{1,2})\s+"
    r"(\d{2}):(\d{2}):(\d{2})\s+"
    r"(\d{4})"
)

MONTHS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
    "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
    "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


# ===========================================================================
# PARSING FUNCTIONS
# ===========================================================================

def read_log(log_path):
    """Read an OpenMOC log file."""
    return log_path.read_text(encoding="utf-8", errors="replace")


def extract_iterations(text):
    """
    Extract arrays of iteration number, k_eff and residual.

    Expected lines look like:
        Iteration 10:  k_eff = 1.012345   res = 1.234E-04 ...
    """
    iterations = []
    keff = []
    residual = []

    for match in ITERATION_RE.finditer(text):
        iterations.append(int(match.group(1)))
        keff.append(float(match.group(2)))
        residual.append(float(match.group(3)))

    if not iterations:
        raise ValueError("No OpenMOC iteration records were found.")

    return iterations, keff, residual


def extract_parameters(text):
    """Extract available discretisation and run parameters."""
    params = {}

    for key, pattern in PARAMETER_PATTERNS.items():
        match = pattern.search(text)

        if not match:
            continue

        value = match.group(1)

        if key in {
            "azim_angles",
            "polar_angles",
            "threads",
            "fsrs",
            "tracks",
        }:
            params[key] = int(value)
        else:
            params[key] = float(value)

    return params


def extract_run_timestamp(text):
    """
    Extract a compact timestamp from:
        Current local time and date: Wed Nov 11 19:55:08 2026

    Returns:
        '20261111_195508'

    Returns None if the date line is absent.
    """
    match = DATE_RE.search(text)

    if not match:
        return None

    month_name, day, hour, minute, second, year = match.groups()

    dt = datetime(
        year=int(year),
        month=MONTHS[month_name],
        day=int(day),
        hour=int(hour),
        minute=int(minute),
        second=int(second),
    )

    return dt.strftime("%Y%m%d_%H%M%S")


# ===========================================================================
# LABEL / FILENAME HELPERS
# ===========================================================================

def compact_number(value):
    """
    Convert a number into a filename-safe compact representation.

    Examples
    --------
    5.0    -> 5
    2.5    -> 2p5
    0.625  -> 0p625
    """
    return f"{value:g}".replace("-", "m").replace(".", "p")


def make_parameter_tag(params):
    """
    Construct a compact filename parameter tag.

    Example:
        a4_p2_s2p5_z5_t4_FSR99177
    """
    pieces = []

    if "azim_angles" in params:
        pieces.append(f"a{params['azim_angles']}")

    if "polar_angles" in params:
        pieces.append(f"p{params['polar_angles']}")

    if "azim_spacing" in params:
        pieces.append(f"s{compact_number(params['azim_spacing'])}")

    if "z_spacing" in params:
        pieces.append(f"z{compact_number(params['z_spacing'])}")

    if "threads" in params:
        pieces.append(f"t{params['threads']}")

    if "fsrs" in params:
        pieces.append(f"FSR{params['fsrs']}")

    return "_".join(pieces) if pieces else "parameters_unknown"


def make_title(params):
    """
    Construct a concise title containing the detected discretisation parameters.
    """
    items = []

    if "azim_angles" in params:
        items.append(f"a={params['azim_angles']}")

    if "polar_angles" in params:
        items.append(f"p={params['polar_angles']}")

    if "azim_spacing" in params:
        items.append(f"s={params['azim_spacing']:g} cm")

    if "z_spacing" in params:
        items.append(f"z={params['z_spacing']:g} cm")

    if "threads" in params:
        items.append(f"threads={params['threads']}")

    if "fsrs" in params:
        items.append(f"FSRs={params['fsrs']:,}")

    if items:
        return "OpenMOC convergence | " + ", ".join(items)

    return "OpenMOC convergence"


# ===========================================================================
# PLOTTING
# ===========================================================================

def create_convergence_plot(
    log_path,
    iterations,
    keff,
    residual,
    params,
    timestamp,
    plots_dir,
):
    """
    Create and save one convergence figure.

    The figure is NOT shown here. All figures are shown together after every
    log file has been processed.
    """

    parameter_tag = make_parameter_tag(params)

    filename_parts = [
        "convergence",
        parameter_tag,
    ]

    if timestamp:
        filename_parts.append(timestamp)

    output_name = "_".join(filename_parts) + ".pdf"
    output_path = plots_dir / output_name

    # -----------------------------------------------------------------------
    # Figure and left axis: k_eff
    # -----------------------------------------------------------------------

    fig, ax_keff = plt.subplots(figsize=(11, 6.5))

    line_keff, = ax_keff.plot(
        iterations,
        keff,
        color="red",
        linewidth=1.7,
        label=r"$k_{\mathrm{eff}}$",
    )

    ax_keff.set_xlabel("Iteration", fontsize=13)
    ax_keff.set_ylabel(
        r"$k_{\mathrm{eff}}$",
        color="red",
        fontsize=13,
    )

    ax_keff.tick_params(
        axis="y",
        labelcolor="red",
    )

    ax_keff.grid(
        True,
        alpha=0.25,
    )

    # -----------------------------------------------------------------------
    # Right axis: residual
    # -----------------------------------------------------------------------

    ax_res = ax_keff.twinx()

    line_res, = ax_res.plot(
        iterations,
        residual,
        color="blue",
        linewidth=1.5,
        label="Residual",
    )

    ax_res.set_ylabel(
        "Source residual",
        color="blue",
        fontsize=13,
    )

    ax_res.tick_params(
        axis="y",
        labelcolor="blue",
    )

    # Residual normally spans several orders of magnitude.
    ax_res.set_yscale("log")

    # -----------------------------------------------------------------------
    # Title
    # -----------------------------------------------------------------------

    ax_keff.set_title(
        make_title(params),
        fontsize=15,
        pad=14,
    )

    # -----------------------------------------------------------------------
    # Combined legend
    # -----------------------------------------------------------------------

    ax_keff.legend(
        [line_keff, line_res],
        [line_keff.get_label(), line_res.get_label()],
        loc="lower right",
        fontsize=11,
    )

    # -----------------------------------------------------------------------
    # Final-value annotation
    # -----------------------------------------------------------------------

    final_text = (
        f"Final iteration: {iterations[-1]}\n"
        f"Final $k_{{eff}}$: {keff[-1]:.6f}\n"
        f"Final residual: {residual[-1]:.3e}"
    )

    # User-preferred annotation position / size
    ax_keff.text(
        0.90,
        0.90,
        final_text,
        transform=ax_keff.transAxes,
        ha="right",
        va="top",
        fontsize=14,
        bbox={"boxstyle": "round", "alpha": 0.75},
    )

    fig.tight_layout()

    # Save before plt.show()
    fig.savefig(
        output_path,
        format="pdf",
        bbox_inches="tight",
    )

    return fig, output_path


# ===========================================================================
# MAIN BATCH WORKFLOW
# ===========================================================================

def main():

    # The script, log/ folder and plots/ folder all live at the same level.
    script_dir = Path(__file__).resolve().parent
    log_dir = script_dir / "log"
    plots_dir = script_dir / "plots"

    # plots/ already exists in your setup, but this makes the script robust.
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not log_dir.exists():
        print(
            f"ERROR: log directory does not exist:\n{log_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Find every .log file directly inside ./log/
    log_files = sorted(
        path for path in log_dir.glob("*.log")
        if path.is_file()
    )

    if not log_files:
        print(
            f"ERROR: no .log files found in:\n{log_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    print()
    print("OpenMOC convergence batch plotting")
    print("==================================")
    print(f"Log directory:   {log_dir}")
    print(f"Plots directory: {plots_dir}")
    print(f"Log files found: {len(log_files)}")
    print()

    figures = []
    successful = 0
    failed = 0

    for index, log_path in enumerate(log_files, start=1):

        print(f"[{index}/{len(log_files)}] {log_path.name}")

        try:
            text = read_log(log_path)

            iterations, keff, residual = extract_iterations(text)
            params = extract_parameters(text)
            timestamp = extract_run_timestamp(text)
            parameter_tag = make_parameter_tag(params)

            filename_parts = [
                "convergence",
                parameter_tag,
            ]

            if timestamp:
                filename_parts.append(timestamp)

            output_name = "_".join(filename_parts) + ".pdf"
            output_path = plots_dir / output_name

            if output_path.exists():
                print(f"    SKIPPED: plot already exists -> {output_path.name}")
                print()
                continue

            fig, output_path = create_convergence_plot(
                log_path=log_path,
                iterations=iterations,
                keff=keff,
                residual=residual,
                params=params,
                timestamp=timestamp,
                plots_dir=plots_dir,
            )

            figures.append(fig)
            successful += 1

            print(f"    iterations     : {len(iterations)}")
            print(f"    final k_eff    : {keff[-1]:.6f}")
            print(f"    final residual : {residual[-1]:.3e}")
            print(f"    PDF             : {output_path.name}")
            print()

        except Exception as exc:
            failed += 1
            print(f"    SKIPPED: {exc}")
            print()

    print("Summary")
    print("-------")
    print(f"Successful plots: {successful}")
    print(f"Skipped logs:     {failed}")
    print()

    if not figures:
        print(
            "No plots could be generated.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        "All PDFs have been saved. "
        "Opening the generated figures now..."
    )

    # Display every generated figure.
    # The program remains open until the plot windows are closed.
    plt.show()


if __name__ == "__main__":
    main()
