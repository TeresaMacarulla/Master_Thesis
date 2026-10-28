"""
Generate material-wise 7-group MGXS with OpenMC and export them in the
OpenMOC HDF5 format expected by openmoc.materialize.load_from_hdf5().

Required input files in this script's directory:
    materials.xml
    geometry.xml
    settings.xml

Outputs in this script's directory:
    materials.h5          -> file consumed directly by OpenMOC
    materials-hdf5.py     -> reproducible Python generator containing the
                             extracted arrays and able to recreate materials.h5

OpenMC run files are kept in:
    mgxs-run/
"""

from pathlib import Path
import h5py
import numpy as np
import openmc
import openmc.mgxs
import shutil

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
RUN_DIR = BASE_DIR / "mgxs-run"

GEOMETRY_XML = BASE_DIR / "geometry.xml"
SETTINGS_XML = BASE_DIR / "settings.xml"
MATERIALS_XML = BASE_DIR / "materials.xml"

OUTPUT_H5 = BASE_DIR / "materials.h5"
OUTPUT_GENERATOR = BASE_DIR / "materials-hdf5.py"

# =============================================================================
# ENERGY GROUP STRUCTURE
# =============================================================================

# OpenMC receives energy boundaries from low to high energy [eV].
# OpenMOC expects arrays ordered by energy-group number, with group 1 fast.
GROUP_EDGES = np.array([
    1.0e-5,
    6.35e-2,
    1.0e1,
    1.0e2,
    1.0e3,
    5.0e5,
    1.0e6,
    2.0e7,
])

ENERGY_GROUPS = openmc.mgxs.EnergyGroups(GROUP_EDGES)
NUM_GROUPS = len(GROUP_EDGES) - 1

# =============================================================================
# CHECK / LOAD INPUT MODEL
# =============================================================================

for path in (GEOMETRY_XML, SETTINGS_XML, MATERIALS_XML):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Place geometry.xml, settings.xml and materials.xml next to this script."
        )

# Load the material objects first, then force the geometry to use those same
# material instances. This avoids having two independent sets of Material objects.
materials = openmc.Materials.from_xml(MATERIALS_XML)
geometry = openmc.Geometry.from_xml(GEOMETRY_XML, materials=materials)
settings = openmc.Settings.from_xml(SETTINGS_XML)

print("\nMaterial temperatures supplied to the OpenMC MGXS model:")
print("-" * 72)
for material in materials:
    print(
        f"ID {material.id:>3} | {material.name!r:<25} | "
        f"T = {material.temperature} K"
    )
print("-" * 72)

# =============================================================================
# CREATE MATERIAL-WISE MGXS LIBRARY + TALLIES
# =============================================================================

mgxs_lib = openmc.mgxs.Library(geometry)
mgxs_lib.energy_groups = ENERGY_GROUPS
mgxs_lib.domain_type = "material"
mgxs_lib.domains = list(geometry.get_all_materials().values())
mgxs_lib.by_nuclide = False
mgxs_lib.mgxs_types = [
    "total",
    "scatter matrix",
    "fission",
    "nu-fission",
    "chi",
]
mgxs_lib.scatter_format = "legendre"
mgxs_lib.legendre_order = 0
mgxs_lib.correction = None
mgxs_lib.build_library()

tallies = openmc.Tallies()
if hasattr(mgxs_lib, "add_to_tallies"):
    mgxs_lib.add_to_tallies(tallies, merge=True)
else:
    mgxs_lib.add_to_tallies_file(tallies, merge=True)

# =============================================================================
# RUN THE EXACT MODEL WITH THE MGXS TALLIES ATTACHED
# =============================================================================

RUN_DIR.mkdir(parents=True, exist_ok=True)

# Remove old MGXS run products so that an old statepoint/tallies.xml cannot be
# accidentally reused.
for pattern in ("statepoint.*.h5", "summary.h5", "tallies.xml", "model.xml"):
    for old_file in RUN_DIR.glob(pattern):
        old_file.unlink()

mgxs_model = openmc.Model(
    geometry=geometry,
    materials=materials,
    settings=settings,
    tallies=tallies,
)

# Export separate XML files into the dedicated run directory.  This guarantees
# that the tallies just created above are included in the actual OpenMC run.
mgxs_model.export_to_xml(directory=RUN_DIR)

print("Starting OpenMC MGXS calculation...")
print(f"Run directory: {RUN_DIR}")
print(f"Number of material domains: {len(mgxs_lib.domains)}")
print(f"Number of energy groups: {NUM_GROUPS}\n")

openmc.run(cwd=RUN_DIR, threads=4)

statepoint_path = RUN_DIR / f"statepoint.{settings.batches}.h5"
if not statepoint_path.exists():
    raise FileNotFoundError(
        f"Expected final statepoint was not found: {statepoint_path}"
    )

# StatePoint autolinks summary.h5 by default when it is present in the same
# directory, so do not call link_with_summary() a second time.
with openmc.StatePoint(statepoint_path) as sp:
    if sp.summary is None:
        summary_path = RUN_DIR / "summary.h5"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"Statepoint was not linked and summary.h5 is missing: {summary_path}"
            )
        sp.link_with_summary(openmc.Summary(summary_path))

    mgxs_lib.load_from_statepoint(sp)

# =============================================================================
# EXTRACTION HELPERS
# =============================================================================

def clean(values):
    """Return finite double-precision NumPy data."""
    return np.nan_to_num(
        np.asarray(values, dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def get_vector(material, xs_type):
    """Return a G-component macroscopic XS vector [cm^-1]."""
    mgxs = mgxs_lib.get_mgxs(material, xs_type)
    values = mgxs.get_xs(
        xs_type="macro",
        order_groups="increasing",
        value="mean",
        squeeze=True,
    )
    return clean(values).reshape(NUM_GROUPS)


def get_scatter(material):
    """Return P0 scattering matrix, rows=incoming and columns=outgoing."""
    mgxs = mgxs_lib.get_mgxs(material, "scatter matrix")
    values = mgxs.get_xs(
        xs_type="macro",
        order_groups="increasing",
        row_column="inout",
        moment=0,
        value="mean",
        squeeze=True,
    )
    return clean(values).reshape(NUM_GROUPS, NUM_GROUPS)


def get_chi(material, sigma_f):
    """Return normalized chi, or zeros for a non-fissionable material."""
    if np.allclose(sigma_f, 0.0):
        return np.zeros(NUM_GROUPS)

    mgxs = mgxs_lib.get_mgxs(material, "chi")
    values = mgxs.get_xs(
        order_groups="increasing",
        value="mean",
        squeeze=True,
    )
    chi = clean(values).reshape(NUM_GROUPS)
    chi[chi < 0.0] = 0.0
    if chi.sum() > 0.0:
        chi /= chi.sum()
    return chi

# =============================================================================
# COLLECT MATERIAL DATA
# =============================================================================

material_data = {}

for material in mgxs_lib.domains:
    sigma_t = get_vector(material, "total")
    sigma_s = get_scatter(material)
    sigma_f = get_vector(material, "fission")
    nu_sigma_f = get_vector(material, "nu-fission")
    chi = get_chi(material, sigma_f)

    material_data[material.name] = {
        "total": sigma_t,
        "scatter matrix": sigma_s.ravel(order="C"),
        "fission": sigma_f,
        "nu-fission": nu_sigma_f,
        "chi": chi,
    }

# OpenMC represents the outside-vessel buffer as true void (fill=None), so it
# does not appear as an MGXS material. OpenMOC nevertheless needs a Material
# object in that region. Add an effectively non-interacting near-void.
material_data["void"] = {
    "total": np.full(NUM_GROUPS, 1.0e-10),
    "scatter matrix": np.zeros(NUM_GROUPS * NUM_GROUPS),
    "fission": np.zeros(NUM_GROUPS),
    "nu-fission": np.zeros(NUM_GROUPS),
    "chi": np.zeros(NUM_GROUPS),
}

# =============================================================================
# WRITE FINAL OPENMOC-COMPATIBLE materials.h5
# =============================================================================

with h5py.File(OUTPUT_H5, "w") as h5:

    # Number of energy groups
    h5.attrs["# groups"] = NUM_GROUPS
    # OpenMOC expects the domain type at the top level
    material_group = h5.create_group("material")

    for material_name, data in material_data.items():
        group = material_group.create_group(material_name)
        group.create_dataset("total", data=data["total"])
        group.create_dataset(
            "scatter matrix",
            data=data["scatter matrix"]
        )
        group.create_dataset(
            "fission",
            data=data["fission"]
        )
        group.create_dataset(
            "nu-fission",
            data=data["nu-fission"]
        )
        group.create_dataset(
            "chi",
            data=data["chi"]
        )

# =============================================================================
# WRITE A REPRODUCIBLE materials-hdf5.py GENERATOR
# =============================================================================

def array_literal(values, indent="        "):
    arr = np.asarray(values, dtype=float).ravel()
    return "[\n" + "\n".join(
        f"{indent}{value:.12E}," for value in arr
    ) + "\n    ]"

lines = [
    '"""Auto-generated OpenMOC material-library writer.\n',
    'Run this file to recreate materials.h5 from the MGXS values extracted\n',
    'by calculate_material_mgxs.py.\n"""\n\n',
    'from pathlib import Path\n',
    'import h5py\n',
    'import numpy as np\n\n',
    'BASE_DIR = Path(__file__).resolve().parent\n',
    'OUTPUT_H5 = BASE_DIR / "materials.h5"\n',
    f'NUM_GROUPS = {NUM_GROUPS}\n\n',
    'materials = {}\n\n',
]

for material_name, data in material_data.items():
    safe_name = repr(material_name)
    lines.append(f"materials[{safe_name}] = {{\n")
    for dataset_name in ("total", "scatter matrix", "fission", "nu-fission", "chi"):
        lines.append(
            f"    {dataset_name!r}: np.array({array_literal(data[dataset_name])}, dtype=float),\n"
        )
    lines.append("}\n\n")

lines.extend([
    'with h5py.File(OUTPUT_H5, "w") as h5:\n',
    '    h5.attrs["# groups"] = NUM_GROUPS\n',
    '    material_group = h5.create_group("material")\n',
    '    for material_name, data in materials.items():\n',
    '        group = material_group.create_group(material_name)\n',
    '        for dataset_name, values in data.items():\n',
    '            group.create_dataset(dataset_name, data=values)\n\n',
    'print(f"Wrote OpenMOC material library: {OUTPUT_H5}")\n',
])

OUTPUT_GENERATOR.write_text("".join(lines), encoding="utf-8")

print("\n" + "=" * 79)
print("MGXS extraction completed successfully.")
print(f"OpenMOC HDF5 library:   {OUTPUT_H5}")
print(f"Reproducible generator: {OUTPUT_GENERATOR}")
print("=" * 79)

# =============================================================================
# REMOVE TEMPORARY OPENMC MGXS RUN DIRECTORY
# =============================================================================

if RUN_DIR.exists():
    shutil.rmtree(RUN_DIR)

print(f"Temporary MGXS run directory removed: {RUN_DIR}")
