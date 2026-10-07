"""
calculate_material_mgxs.py

Calculate and print the material-wise multigroup macroscopic cross sections
needed to build an OpenMOC-style HDF5 material library by hand.

The script DOES NOT create the final OpenMOC HDF5 file. It runs an OpenMC
continuous-energy calculation with MGXS tallies, then prints Python-ready
arrays for each material:

    sigma_t
    sigma_s
    sigma_f
    nu_sigma_f
    chi

It also writes the same output to:
    mgxs_arrays.txt

Required files in the same directory:
    materials.py
    geometry.xml
    settings.xml

The script imports the material definitions from materials.py, so the
material temperatures defined there are used unless a cell temperature
in geometry.xml overrides them.

Run from an environment in which the OpenMC Python package and the OpenMC
executable are installed, for example:

    python calculate_material_mgxs.py
"""

from pathlib import Path
import sys

import numpy as np
import openmc
import openmc.mgxs

# Import the existing OpenMC materials exactly as defined by the model.
# NOTE: materials.py currently exports materials.xml as a side effect.
from materials import materials


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

GEOMETRY_XML = BASE_DIR / "geometry.xml"
SETTINGS_XML = BASE_DIR / "settings.xml"

RUN_DIR = BASE_DIR / "mgxs-run"
OUTPUT_TXT = BASE_DIR / "mgxs_arrays.txt"


# =============================================================================
# ENERGY GROUP STRUCTURE
# =============================================================================
#
# Provisional 7-group C5G7 structure [eV].
#
# OpenMC stores the boundaries from LOW energy to HIGH energy.
# When the results are printed below, the arrays are ordered as:
#
#     Group 1 = highest-energy group
#     ...
#     Group 7 = lowest-energy group
#
# which matches the usual OpenMOC/C5G7 ordering.
# =============================================================================

GROUP_EDGES = np.array([ 1.0e-5, 6.35e-2, 1.0e1, 1.0e2, 1.0e3, 5.0e5, 1.0e6, 2.0e7,])
ENERGY_GROUPS = openmc.mgxs.EnergyGroups(GROUP_EDGES)
NUM_GROUPS = len(GROUP_EDGES) - 1


# =============================================================================
# CHECK INPUT FILES
# =============================================================================

for path in (GEOMETRY_XML, SETTINGS_XML):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Place geometry.xml and settings.xml in the same directory "
            "as this script."
        )


# =============================================================================
# LOAD THE EXISTING OPENMC MODEL
# =============================================================================
#
# Geometry is loaded using the material objects imported from materials.py.
# Therefore the temperatures, densities, compositions and S(alpha,beta)
# definitions in materials.py are retained.
# =============================================================================

geometry = openmc.Geometry.from_xml( GEOMETRY_XML, materials=materials,)
settings = openmc.Settings.from_xml( SETTINGS_XML )


# =============================================================================
# PRINT MATERIAL TEMPERATURES USED
# =============================================================================

print("\nMaterial temperatures supplied to the OpenMC model:")
print("-" * 65)

for material in materials:
    print(
        f"ID {material.id:>3} | "
        f"{material.name!r:<25} | "
        f"T = {material.temperature} K"
    )

print("-" * 65)
print(
    "A cell-specific temperature in geometry.xml, if present, "
    "would override the material temperature.\n"
)


# =============================================================================
# CREATE MATERIAL-WISE MGXS LIBRARY
# =============================================================================
#
# These five quantities correspond to the arrays used in the OpenMOC
# c5g7-mgxs-hdf5.py style:
#
#   total          -> sigma_t
#   scatter matrix -> sigma_s
#   fission        -> sigma_f
#   nu-fission     -> nu_sigma_f
#   chi            -> chi
#
# P0 scattering is used here so that the scattering result is one
# G x G matrix, matching the simple OpenMOC example.
# =============================================================================

mgxs_lib = openmc.mgxs.Library(geometry)
mgxs_lib.energy_groups = ENERGY_GROUPS
mgxs_lib.domain_type = "material"

# Only calculate cross sections for materials actually present in the geometry.
mgxs_lib.domains = list( geometry.get_all_materials().values() )
mgxs_lib.by_nuclide = False
mgxs_lib.mgxs_types = [ "total", "scatter matrix", "fission", "nu-fission", "chi", ]
mgxs_lib.scatter_format = "legendre"
mgxs_lib.legendre_order = 0

# Do not apply a P0 transport correction for this first extraction.
mgxs_lib.correction = None

mgxs_lib.build_library()


# =============================================================================
# CREATE THE TALLIES REQUIRED FOR THE MGXS CALCULATION
# =============================================================================

tallies = openmc.Tallies()

# Newer OpenMC versions use add_to_tallies(); older versions used
# add_to_tallies_file(). This keeps the script compatible with both.
if hasattr(mgxs_lib, "add_to_tallies"):
    mgxs_lib.add_to_tallies(
        tallies,
        merge=True
    )
else:
    mgxs_lib.add_to_tallies_file(
        tallies,
        merge=True
    )


# =============================================================================
# BUILD A SEPARATE OPENMC RUN DIRECTORY
# =============================================================================
#
# This keeps the MGXS calculation separate from the original model files.
# =============================================================================

RUN_DIR.mkdir(
    parents=True,
    exist_ok=True
)

mgxs_model = openmc.Model(
    geometry=geometry,
    materials=materials,
    settings=settings,
    tallies=tallies,
)

mgxs_model.export_to_xml(
    directory=RUN_DIR
)


# =============================================================================
# RUN OPENMC
# =============================================================================

print("Starting OpenMC MGXS calculation...")
print(f"Run directory: {RUN_DIR}")
print(f"Number of material domains: {len(mgxs_lib.domains)}")
print(f"Number of energy groups: {NUM_GROUPS}")
print()

openmc.run(
    cwd=RUN_DIR
)


# =============================================================================
# LOAD STATEPOINT + MGXS RESULTS
# =============================================================================

statepoint_path = RUN_DIR / f"statepoint.{settings.batches}.h5"
summary_path = RUN_DIR / "summary.h5"

if not statepoint_path.exists():
    raise FileNotFoundError(
        f"Expected OpenMC statepoint was not found: {statepoint_path}"
    )

if not summary_path.exists():
    raise FileNotFoundError(
        f"Expected OpenMC summary file was not found: {summary_path}"
    )

with openmc.StatePoint(statepoint_path) as sp:

    summary = openmc.Summary(summary_path)
    sp.link_with_summary(summary)

    mgxs_lib.load_from_statepoint(sp)


# =============================================================================
# EXTRACTION HELPERS
# =============================================================================

def clean(values):
    """
    Convert output to finite floating-point values.

    Non-fissionable materials can produce undefined chi values (0/0).
    These are replaced by zero.
    """

    return np.nan_to_num(
        np.asarray(values, dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def get_vector(material, xs_type):
    """
    Return a G-component material-wise macroscopic cross-section vector
    in cm^-1.
    """

    mgxs = mgxs_lib.get_mgxs(
        material,
        xs_type
    )

    values = mgxs.get_xs(
        xs_type="macro",
        order_groups="increasing",
        value="mean",
        squeeze=True,
    )

    return clean(values).reshape(NUM_GROUPS)


def get_scatter(material):
    """
    Return the P0 G x G scattering matrix in cm^-1.

    Rows = incoming groups
    Columns = outgoing groups
    """

    mgxs = mgxs_lib.get_mgxs(
        material,
        "scatter matrix"
    )

    values = mgxs.get_xs(
        xs_type="macro",
        order_groups="increasing",
        row_column="inout",
        moment=0,
        value="mean",
        squeeze=True,
    )

    return clean(values).reshape(
        NUM_GROUPS,
        NUM_GROUPS
    )


def get_chi(material, sigma_f):
    """
    Return the G-component fission spectrum.

    For non-fissionable materials, return zeros.
    """

    if np.allclose(sigma_f, 0.0):
        return np.zeros(NUM_GROUPS)

    mgxs = mgxs_lib.get_mgxs(
        material,
        "chi"
    )

    values = mgxs.get_xs(
        order_groups="increasing",
        value="mean",
        squeeze=True,
    )

    chi = clean(values).reshape(NUM_GROUPS)

    # Remove tiny Monte Carlo numerical artefacts and normalize.
    chi[chi < 0.0] = 0.0

    if chi.sum() > 0.0:
        chi /= chi.sum()

    return chi


# =============================================================================
# FORMATTING HELPERS
# =============================================================================

def format_vector(name, values):
    """Return a copy/paste-ready NumPy vector definition."""

    elements = ", ".join(
        f"{value:.8E}"
        for value in values
    )

    return (
        f"{name} = np.array([\n"
        f"    {elements}\n"
        f"])\n"
    )


def format_scatter(values):
    """
    Return the G x G scatter matrix as the flattened vector used by the
    OpenMOC C5G7-style HDF5 scripts.
    """

    flat = values.ravel(order="C")

    lines = []

    for start in range(0, flat.size, NUM_GROUPS):
        row = flat[start:start + NUM_GROUPS]

        lines.append(
            "    "
            + ", ".join(f"{value:.8E}" for value in row)
        )

    return (
        "sigma_s = np.array([\n"
        + ",\n".join(lines)
        + "\n])\n"
    )


# =============================================================================
# PRINT / SAVE ALL MATERIAL ARRAYS
# =============================================================================

output_blocks = []

for material in mgxs_lib.domains:

    sigma_t = get_vector(
        material,
        "total"
    )

    sigma_s_matrix = get_scatter(
        material
    )

    sigma_f = get_vector(
        material,
        "fission"
    )

    nu_sigma_f = get_vector(
        material,
        "nu-fission"
    )

    chi = get_chi(
        material,
        sigma_f
    )

    block = []

    block.append(
        "\n"
        + "#" * 79
        + "\n"
        + f"# MATERIAL ID {material.id}: {material.name}\n"
        + f"# Temperature = {material.temperature} K\n"
        + "#" * 79
        + "\n"
    )

    block.append(
        format_vector(
            "sigma_t",
            sigma_t
        )
    )

    block.append(
        format_scatter(
            sigma_s_matrix
        )
    )

    block.append(
        format_vector(
            "sigma_f",
            sigma_f
        )
    )

    block.append(
        format_vector(
            "nu_sigma_f",
            nu_sigma_f
        )
    )

    block.append(
        format_vector(
            "chi",
            chi
        )
    )

    output_blocks.append(
        "\n".join(block)
    )


full_output = "\n".join(
    output_blocks
)

print(full_output)


OUTPUT_TXT.write_text(
    full_output,
    encoding="utf-8"
)

print()
print("=" * 79)
print("MGXS extraction completed.")
print(f"Copy/paste-ready arrays were also written to:")
print(f"    {OUTPUT_TXT}")
print("=" * 79)
