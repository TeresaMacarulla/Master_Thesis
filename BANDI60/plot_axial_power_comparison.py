"""
Compare normalized axial power distributions from OpenMC and OpenMOC.

Expected directory structure
----------------------------
BANDI60/
├── plot_axial_power_comparison.py
├── OpenMC_model/
│   └── log/
│       └── axial_power_openmc_*.csv
├── OpenMOC_model/
│   └── log/
│       └── fsr_fission_*.csv
└── plots/
    └── axial_power_comparison.pdf

OpenMC expected columns
-----------------------
z_lower_cm,z_upper_cm,z_mid_cm,core_height_cm,
fission_rate,std_dev,normalized_axial_power

OpenMOC expected columns
------------------------
fsr_id,x_cm,y_cm,z_cm,volume_cm3,fission_rate

The plot orientation follows Kim et al. (2024), Fig. 20:
    x-axis: normalized axial power
    y-axis: core height [cm]
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# SETTINGS
# =============================================================================

ACTIVE_Z_MIN = -100.0  # cm
ACTIVE_Z_MAX = 100.0   # cm

OPENMC_PATTERN = "axial_power_openmc_*.csv"
OPENMOC_PATTERN = "fsr_fission_*.csv"

OUTPUT_FILENAME = "axial_power_comparison.pdf"


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

OPENMC_LOG_DIR = BASE_DIR / "OpenMC_model" / "log"
OPENMOC_LOG_DIR = BASE_DIR / "OpenMOC_model" / "log"

PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PDF = PLOTS_DIR / OUTPUT_FILENAME


# =============================================================================
# HELPERS
# =============================================================================

def find_single_file(folder, pattern, label):
    """Return the only matching file in folder; fail clearly otherwise."""
    if not folder.exists():
        raise FileNotFoundError(f"{label} log folder does not exist:\n{folder}")

    matches = sorted(folder.glob(pattern))

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No {label} file matching '{pattern}' was found in:\n{folder}"
        )

    if len(matches) > 1:
        files = "\n".join(f"  - {path.name}" for path in matches)
        raise RuntimeError(
            f"Expected exactly one {label} file, but found {len(matches)}:\n{files}"
        )

    return matches[0]


def load_named_csv(path):
    """Load a CSV with a header row and preserve column names."""
    data = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=float,
        encoding="utf-8",
    )
    return np.atleast_1d(data)


def require_columns(data, required, path):
    """Check that all required named columns are present."""
    available = set(data.dtype.names or ())
    missing = [name for name in required if name not in available]

    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: {missing}\n"
            f"Available columns: {sorted(available)}"
        )


def infer_bin_widths_from_centers(z, lower_bound, upper_bound):
    """
    Estimate the axial interval represented by each sorted z-center.

    Interior boundaries are halfway between neighboring centers.
    The first and last boundaries are fixed at the active-core limits.
    """
    z = np.asarray(z, dtype=float)

    if z.size < 2:
        raise ValueError(
            "At least two axial z levels are required to infer an axial profile."
        )

    if np.any(np.diff(z) <= 0.0):
        raise ValueError("Axial z coordinates must be strictly increasing.")

    edges = np.empty(z.size + 1, dtype=float)
    edges[0] = lower_bound
    edges[-1] = upper_bound
    edges[1:-1] = 0.5 * (z[:-1] + z[1:])

    widths = np.diff(edges)

    if np.any(widths <= 0.0):
        raise ValueError(
            "Could not construct positive axial intervals from OpenMOC z-centers."
        )

    return widths


def normalize_integrated_axial_rates(rates, widths):
    """
    Convert integrated fission rate per axial interval into relative axial power.

    First divide by interval width to obtain a quantity proportional to
    power per unit core height. Then normalize by the height-weighted mean.
    """
    rates = np.asarray(rates, dtype=float)
    widths = np.asarray(widths, dtype=float)

    if rates.shape != widths.shape:
        raise ValueError("rates and widths must have identical shapes.")

    if np.any(widths <= 0.0):
        raise ValueError("All axial widths must be positive.")

    axial_rate_density = rates / widths
    weighted_mean = np.sum(axial_rate_density * widths) / np.sum(widths)

    if not np.isfinite(weighted_mean) or weighted_mean <= 0.0:
        raise ValueError("Cannot normalize: non-positive mean fission rate.")

    return axial_rate_density / weighted_mean


# =============================================================================
# OPENMC
# =============================================================================

def read_openmc_profile(path):
    """Read the already axially binned OpenMC fission-rate file."""
    data = load_named_csv(path)

    require_columns(
        data,
        [
            "z_lower_cm",
            "z_upper_cm",
            "z_mid_cm",
            "core_height_cm",
            "fission_rate",
        ],
        path,
    )

    order = np.argsort(data["core_height_cm"])

    core_height = np.asarray(data["core_height_cm"][order], dtype=float)
    fission_rate = np.asarray(data["fission_rate"][order], dtype=float)

    widths = (
        np.asarray(data["z_upper_cm"][order], dtype=float)
        - np.asarray(data["z_lower_cm"][order], dtype=float)
    )

    normalized_power = normalize_integrated_axial_rates(
        fission_rate,
        widths,
    )

    # Optional consistency check with the value already stored by OpenMC.
    if "normalized_axial_power" in (data.dtype.names or ()):
        stored = np.asarray(data["normalized_axial_power"][order], dtype=float)
        max_difference = np.nanmax(np.abs(stored - normalized_power))

        if max_difference > 1.0e-8:
            print(
                "WARNING: recomputed OpenMC normalization differs from the "
                f"stored normalized_axial_power column by up to "
                f"{max_difference:.3e}."
            )

    return core_height, normalized_power


# =============================================================================
# OPENMOC
# =============================================================================

def read_openmoc_profile(path):
    """
    Convert the FSR-wise OpenMOC output into an axial power profile.

    FSR fission rates are summed for FSRs sharing the same axial centroid.
    """
    data = load_named_csv(path)

    require_columns(
        data,
        [
            "fsr_id",
            "z_cm",
            "fission_rate",
        ],
        path,
    )

    z = np.asarray(data["z_cm"], dtype=float)
    fission_rate = np.asarray(data["fission_rate"], dtype=float)

    finite = np.isfinite(z) & np.isfinite(fission_rate)

    active = (
        finite
        & (z >= ACTIVE_Z_MIN)
        & (z <= ACTIVE_Z_MAX)
    )

    z = z[active]
    fission_rate = fission_rate[active]

    if z.size == 0:
        raise ValueError(
            "No OpenMOC FSR centroids were found inside the active core."
        )

    # Group numerically identical axial centroid levels.
    z_group = np.round(z, decimals=8)

    unique_z, inverse = np.unique(z_group, return_inverse=True)

    axial_fission_rate = np.zeros(unique_z.size, dtype=float)
    np.add.at(axial_fission_rate, inverse, fission_rate)

    order = np.argsort(unique_z)
    unique_z = unique_z[order]
    axial_fission_rate = axial_fission_rate[order]

    if unique_z.size < 2:
        raise RuntimeError(
            "\nThe OpenMOC file contains only one active-core axial FSR "
            f"centroid level (z = {unique_z[0]:g} cm).\n"
            "That solution does not contain enough independent axial FSR "
            "information to construct an axial power distribution."
        )

    widths = infer_bin_widths_from_centers(
        unique_z,
        ACTIVE_Z_MIN,
        ACTIVE_Z_MAX,
    )

    normalized_power = normalize_integrated_axial_rates(
        axial_fission_rate,
        widths,
    )

    core_height = unique_z - ACTIVE_Z_MIN

    return core_height, normalized_power, widths


# =============================================================================
# MAIN
# =============================================================================

def main():
    try:
        openmc_file = find_single_file(
            OPENMC_LOG_DIR,
            OPENMC_PATTERN,
            "OpenMC axial-power",
        )

        openmoc_file = find_single_file(
            OPENMOC_LOG_DIR,
            OPENMOC_PATTERN,
            "OpenMOC FSR-fission",
        )

        print("Files selected")
        print("--------------")
        print(f"OpenMC : {openmc_file}")
        print(f"OpenMOC: {openmoc_file}")
        print()

        mc_height, mc_power = read_openmc_profile(openmc_file)
        moc_height, moc_power, moc_widths = read_openmoc_profile(openmoc_file)

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    moc_label = openmoc_file.stem
    prefix = "fsr_fission_"
    if moc_label.startswith(prefix):
        moc_label = moc_label[len(prefix):]

    print("Axial profile summary")
    print("---------------------")
    print(f"OpenMC points : {len(mc_height)}")
    print(f"OpenMOC levels: {len(moc_height)}")
    print(
        "OpenMOC inferred axial widths [cm]: "
        f"min={moc_widths.min():.6g}, "
        f"max={moc_widths.max():.6g}"
    )
    print()

    fig, ax = plt.subplots(figsize=(7.2, 7.2))

    # Match Kim et al. Fig. 20 orientation:
    # x = normalized axial power, y = core height
    ax.plot(
        mc_power,
        mc_height,
        marker="o",
        markersize=4,
        linewidth=1.7,
        label="OpenMC",
    )

    ax.plot(
        moc_power,
        moc_height,
        marker="s",
        markersize=4,
        linewidth=1.7,
        label=f"OpenMOC ({moc_label})",
    )

    ax.set_xlabel("Normalized axial power", fontsize=13)
    ax.set_ylabel("Core Height [cm]", fontsize=13)
    ax.set_ylim(0.0, ACTIVE_Z_MAX - ACTIVE_Z_MIN)
    ax.grid(True, alpha=0.3)
    ax.set_title("BANDI-60 axial power distribution", fontsize=14)
    ax.legend(fontsize=10)

    fig.tight_layout()

    fig.savefig(
        OUTPUT_PDF,
        format="pdf",
        bbox_inches="tight",
    )

    print(f"PDF written to: {OUTPUT_PDF}")

    plt.show()


if __name__ == "__main__":
    main()
