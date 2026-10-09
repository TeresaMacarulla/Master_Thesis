"""
geometry.py
===========

OpenMOC geometry for the Pyrex-loaded BANDI-60 active core.

This geometry is based on:
  - the previously developed OpenMC BANDI-60 geometry (Maia),
  - Kim et al. (2020),
  - Kim et al. (2022),
  - Kim et al. (2024).

Scope
-----
This file models the ACTIVE CORE neutronics geometry:
  * 52 Westinghouse-type 17x17 fuel assemblies
  * 4.95 wt.% UO2 fuel material (cross sections supplied externally)
  * five Pyrex BA assembly types: 5, 10, 25, 35, 40 wt.% B2O3
  * 24 Pyrex BA pins per fuel assembly
  * 24 guide-tube positions + 1 central instrument-tube position
  * 200 cm active height
  * 21.50 cm assembly pitch
  * 1.26 cm pin pitch

The outer active-core box is assigned vacuum boundary conditions.

Control rods are also NOT inserted in this first geometry. This corresponds
to an all-rods-out active-core model.

"""

import openmoc
import openmoc.log as log
import openmoc.materialize as materialize
import openmoc.plotter as plotter
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from pathlib import Path
from matplotlib.colors import ListedColormap


# =============================================================================
# USER SETTINGS
# =============================================================================

MGXS_FILENAME = "materials.h5"
MGXS_DIRECTORY = "."

log.set_log_level("NORMAL")


# =============================================================================
# GEOMETRIC DATA [cm]
# =============================================================================

# Fuel / Pyrex pin radii
R_FUEL = 0.4096
R_GAP = 0.4180
R_CLAD = 0.4750

# Lattice dimensions
PIN_PITCH = 1.26
ASSEMBLY_PITCH = 21.50
ACTIVE_HEIGHT = 200.0

N_PIN_X = 17
N_PIN_Y = 17

N_CORE_X = 8
N_CORE_Y = 8

# 17 * 1.26 = 21.42 cm, while the published FA pitch is 21.50 cm.
# Therefore 0.08 cm remains between neighboring pin lattices.
# We assign half of this gap (0.04 cm) to each side of every assembly.
PIN_LATTICE_WIDTH = N_PIN_X * PIN_PITCH
ASSEMBLY_HALF_GAP = (ASSEMBLY_PITCH - PIN_LATTICE_WIDTH) / 2.0

# Guide-tube dimensions inferred from the control-rod radial configuration
# reported by Kim et al. (2024). For the all-rods-out model, the region
# inside the guide tube is treated as water and the outer tube as Zircaloy-4.
R_GUIDE_INNER = 0.561
R_GUIDE_OUTER = 0.602


# =============================================================================
# LOAD MULTIGROUP MATERIALS
# =============================================================================

log.py_printf("NORMAL", "Importing BANDI-60 materials from HDF5...")

materials = materialize.load_from_hdf5(
    filename=MGXS_FILENAME,
    directory=MGXS_DIRECTORY
)

UO2 = materials['UO2']
helium = materials['helium']
zircaloy4 = materials['zircaloy4']
water = materials['water']
pyrex5 = materials['pyrex5']
pyrex10 = materials['pyrex10']
pyrex25 = materials['pyrex25']
pyrex35 = materials['pyrex35']
pyrex40 = materials['pyrex40']
Steel_SA508 = materials['Steel_SA508']
Steel_14404 = materials['Steel_14404']


# =============================================================================
# BASIC PIN-CELL UNIVERSES
# =============================================================================

log.py_printf("NORMAL", "Creating pin-cell universes...")


def make_fuel_pin_universe():
    """UO2 pellet + He gap + Zircaloy-4 cladding + water moderator."""

    fuel_surface = openmoc.ZCylinder(x=0.0, y=0.0, radius=R_FUEL, name="fuel radius")
    gap_surface = openmoc.ZCylinder(x=0.0, y=0.0, radius=R_GAP, name="gap radius")
    clad_surface = openmoc.ZCylinder(x=0.0, y=0.0, radius=R_CLAD, name="clad radius")

    fuel = openmoc.Cell(name="UO2 fuel")
    fuel.setFill(UO2)
    fuel.addSurface(halfspace=-1, surface=fuel_surface)

    gap = openmoc.Cell(name="fuel He gap")
    gap.setFill(helium)
    gap.addSurface(halfspace=+1, surface=fuel_surface)
    gap.addSurface(halfspace=-1, surface=gap_surface)

    clad = openmoc.Cell(name="fuel Zircaloy-4 cladding")
    clad.setFill(zircaloy4)
    clad.addSurface(halfspace=+1, surface=gap_surface)
    clad.addSurface(halfspace=-1, surface=clad_surface)

    moderator = openmoc.Cell(name="fuel-pin moderator")
    moderator.setFill(water)
    moderator.addSurface(halfspace=+1, surface=clad_surface)

    universe = openmoc.Universe(name="fuel pin universe")
    universe.addCell(fuel)
    universe.addCell(gap)
    universe.addCell(clad)
    universe.addCell(moderator)

    return universe


def make_pyrex_pin_universe(pyrex_material, concentration):
    """
    Pyrex BA rod with the same published pellet/gap/cladding radii as
    the UO2 pin.

    The existing OpenMC model uses helium in the BA gap. The 2022 paper also
    labels this gap as He, whereas the 2024 figure labels it as air. This
    implementation follows the existing OpenMC model and the 2022 paper.
    """

    pyrex_surface = openmoc.ZCylinder(
        x=0.0,
        y=0.0,
        radius=R_FUEL,
        name="Pyrex {}% radius".format(concentration)
    )
    gap_surface = openmoc.ZCylinder(
        x=0.0,
        y=0.0,
        radius=R_GAP,
        name="Pyrex {}% gap radius".format(concentration)
    )
    clad_surface = openmoc.ZCylinder(
        x=0.0,
        y=0.0,
        radius=R_CLAD,
        name="Pyrex {}% clad radius".format(concentration)
    )

    absorber = openmoc.Cell(
        name="Pyrex {} wt.% B2O3".format(concentration)
    )
    absorber.setFill(pyrex_material)
    absorber.addSurface(halfspace=-1, surface=pyrex_surface)

    gap = openmoc.Cell(
        name="Pyrex {}% He gap".format(concentration)
    )
    gap.setFill(helium)
    gap.addSurface(halfspace=+1, surface=pyrex_surface)
    gap.addSurface(halfspace=-1, surface=gap_surface)

    clad = openmoc.Cell(
        name="Pyrex {}% Zircaloy-4 cladding".format(concentration)
    )
    clad.setFill(zircaloy4)
    clad.addSurface(halfspace=+1, surface=gap_surface)
    clad.addSurface(halfspace=-1, surface=clad_surface)

    moderator = openmoc.Cell(
        name="Pyrex {}% moderator".format(concentration)
    )
    moderator.setFill(water)
    moderator.addSurface(halfspace=+1, surface=clad_surface)

    universe = openmoc.Universe(
        name="Pyrex {}% pin universe".format(concentration)
    )
    universe.addCell(absorber)
    universe.addCell(gap)
    universe.addCell(clad)
    universe.addCell(moderator)

    return universe


def make_water_universe(name):
    """Homogeneous water universe."""
    cell = openmoc.Cell(name=name + " cell")
    cell.setFill(water)

    universe = openmoc.Universe(name=name)
    universe.addCell(cell)

    return universe


def make_guide_tube_universe(name):
    """
    All-rods-out guide/instrument tube approximation.

    The supplied papers identify 24 guide tubes and one instrument tube.
    Separate instrument-tube dimensions are not reported, so the central
    instrument tube is provisionally represented using the same water-filled
    Zircaloy tube geometry as the guide tubes.
    """

    inner_surface = openmoc.ZCylinder(
        x=0.0,
        y=0.0,
        radius=R_GUIDE_INNER,
        name=name + " inner radius"
    )

    outer_surface = openmoc.ZCylinder(
        x=0.0,
        y=0.0,
        radius=R_GUIDE_OUTER,
        name=name + " outer radius"
    )

    inner_water = openmoc.Cell(name=name + " inner water")
    if name=="guide tube":
        inner_water.setFill(water)
    elif name=="instrument tube":
        inner_water.setFill(zircaloy4)
    inner_water.addSurface(halfspace=-1, surface=inner_surface)

    tube_wall = openmoc.Cell(name=name + " Zircaloy-4 wall")
    tube_wall.setFill(zircaloy4)
    tube_wall.addSurface(halfspace=+1, surface=inner_surface)
    tube_wall.addSurface(halfspace=-1, surface=outer_surface)

    outer_water = openmoc.Cell(name=name + " outer moderator")
    outer_water.setFill(water)
    outer_water.addSurface(halfspace=+1, surface=outer_surface)

    universe = openmoc.Universe(name=name + " universe")
    universe.addCell(inner_water)
    universe.addCell(tube_wall)
    universe.addCell(outer_water)

    return universe


fuel_pin = make_fuel_pin_universe()

pyrex_pin5 = make_pyrex_pin_universe(pyrex5, 5)
pyrex_pin10 = make_pyrex_pin_universe(pyrex10, 10)
pyrex_pin25 = make_pyrex_pin_universe(pyrex25, 25)
pyrex_pin35 = make_pyrex_pin_universe(pyrex35, 35)
pyrex_pin40 = make_pyrex_pin_universe(pyrex40, 40)

guide_tube = make_guide_tube_universe("guide tube")
instrument_tube = make_guide_tube_universe("instrument tube")

water_universe = make_water_universe("water")
water_assembly_universe = make_water_universe("water assembly position")


# =============================================================================
# 17 x 17 FUEL-ASSEMBLY PIN MAP
# =============================================================================
#
# Indices are zero-based: (row, column).
#
# The lists below are the same positions used in the supplied OpenMC model.
# They reproduce:
#   * 24 Pyrex pins
#   * 24 guide tubes
#   * 1 central instrument tube
#   * 240 fuel pins
# =============================================================================

PYREX_HALF_POSITIONS = [
    (1, 5), (1, 11),
    (2, 2), (2, 14),
    (3, 8),
    (4, 4), (4, 12),
    (5, 1), (5, 15),
    (6, 6), (6, 10),
    (8, 1), (8, 15),
]

TUBE_HALF_POSITIONS = [
    (2, 5), (2, 8), (2, 11),
    (3, 3), (3, 13),
    (5, 2), (5, 5), (5, 8), (5, 11), (5, 14),
    (8, 2), (8, 5), (8, 8), (8, 11), (8, 14),
]


def add_180_degree_symmetry(positions):
    """Return each position together with its 180-degree symmetric partner."""
    result = set()

    for row, col in positions:
        result.add((row, col))
        result.add((16 - row, 16 - col))

    return result


PYREX_POSITIONS = add_180_degree_symmetry(PYREX_HALF_POSITIONS)
TUBE_POSITIONS = add_180_degree_symmetry(TUBE_HALF_POSITIONS)

INSTRUMENT_POSITION = (8, 8)
GUIDE_POSITIONS = TUBE_POSITIONS - {INSTRUMENT_POSITION}

assert len(PYREX_POSITIONS) == 24
assert len(GUIDE_POSITIONS) == 24
assert INSTRUMENT_POSITION in TUBE_POSITIONS


# =============================================================================
# CREATE FIVE PYREX FUEL-ASSEMBLY TYPES
# =============================================================================

log.py_printf("NORMAL", "Creating five BANDI-60 Pyrex assembly types...")


def make_assembly(concentration):
    """
    Build one 17x17 Pyrex-loaded BANDI-60 fuel assembly.

    A thin water border is included so that:
        17 * 1.26 cm + 2 * 0.04 cm = 21.50 cm
    reproducing the published assembly pitch.
    """

    # Select the Pyrex pin universe corresponding to this FA type
    if concentration == 5:
        selected_pyrex_pin = pyrex_pin5
    elif concentration == 10:
        selected_pyrex_pin = pyrex_pin10
    elif concentration == 25:
        selected_pyrex_pin = pyrex_pin25
    elif concentration == 35:
        selected_pyrex_pin = pyrex_pin35
    elif concentration == 40:
        selected_pyrex_pin = pyrex_pin40

    # First build the physical 17x17 pin map.
    pin_map = []

    for row in range(N_PIN_Y):

        current_row = []

        for col in range(N_PIN_X):

            position = (row, col)

            if position in PYREX_POSITIONS:
                universe = selected_pyrex_pin

            elif position == INSTRUMENT_POSITION:
                universe = instrument_tube

            elif position in GUIDE_POSITIONS:
                universe = guide_tube

            else:
                universe = fuel_pin

            current_row.append(universe)

        pin_map.append(current_row)

    # Add a 0.04-cm moderator strip around the 17x17 pin lattice.
    #
    # OpenMOC supports non-uniform lattice widths. The complete assembly
    # therefore has 19 x 19 lattice cells:
    #
    #     [0.04] + 17*[1.26] + [0.04] = 21.50 cm
    #
    widths_x = ([ASSEMBLY_HALF_GAP] + [PIN_PITCH] * N_PIN_X + [ASSEMBLY_HALF_GAP])
    widths_y = ([ASSEMBLY_HALF_GAP] + [PIN_PITCH] * N_PIN_Y + [ASSEMBLY_HALF_GAP])

    full_map = []

    full_map.append([water_universe] * (N_PIN_X + 2))
    for row in pin_map:
        full_map.append([water_universe] + row + [water_universe])
    full_map.append([water_universe] * (N_PIN_X + 2))

    assembly_lattice = openmoc.Lattice(
        name="{}% Pyrex FA lattice".format(concentration)
    )

    assembly_lattice.setWidths(widths_x, widths_y, [ACTIVE_HEIGHT])
    assembly_lattice.setUniverses([full_map])
    assembly_cell = openmoc.Cell(name="{}% Pyrex FA lattice cell".format(concentration))
    assembly_cell.setFill(assembly_lattice)
    assembly_universe = openmoc.Universe(name="{}% Pyrex FA".format(concentration))
    assembly_universe.addCell(assembly_cell)

    return assembly_universe


assemblies = {
    5: make_assembly(5),
    10: make_assembly(10),
    25: make_assembly(25),
    35: make_assembly(35),
    40: make_assembly(40),
}


# =============================================================================
# 8 x 8 CORE LOADING PATTERN
# =============================================================================
#
# The 52 active assembly locations reproduce the published BANDI-60 Pyrex
# loading pattern:
#
#             5   10  10   5
#         10  25  25  25  25  10
#      5  25  35  35  35  35  25   5
#     10  25  35  40  40  35  25  10
#     10  25  35  40  40  35  25  10
#      5  25  35  35  35  35  25   5
#         10  25  25  25  25  10
#             5   10  10   5
#
# Empty positions in the surrounding 8x8 rectangular lattice are modeled
# as water. 
# =============================================================================

CORE_MAP = [
    [None, None, 5, 10, 10, 5, None, None],
    [None, 10, 25, 25, 25, 25, 10, None],
    [5, 25, 35, 35, 35, 35, 25, 5],
    [10, 25, 35, 40, 40, 35, 25, 10],
    [10, 25, 35, 40, 40, 35, 25, 10],
    [5, 25, 35, 35, 35, 35, 25, 5],
    [None, 10, 25, 25, 25, 25, 10, None],
    [None, None, 5, 10, 10, 5, None, None],
]


assembly_counts = {
    concentration: sum(
        row.count(concentration)
        for row in CORE_MAP
    )
    for concentration in (5, 10, 25, 35, 40)
}

assert assembly_counts == {5: 8, 10: 12, 25: 16, 35: 12, 40: 4,}

assert sum(assembly_counts.values()) == 52


core_universes = []

for row in CORE_MAP:

    core_row = []

    for concentration in row:

        if concentration is None:
            core_row.append(water_assembly_universe)

        else:
            core_row.append(assemblies[concentration])

    core_universes.append(core_row)


core_lattice = openmoc.Lattice(name="BANDI-60 8x8 core lattice")
core_lattice.setWidth(width_x=ASSEMBLY_PITCH, width_y=ASSEMBLY_PITCH, width_z=ACTIVE_HEIGHT)
core_lattice.setUniverses([core_universes])


# =============================================================================
# ROOT CORE + STEEL REGIONS
# =============================================================================

CORE_WIDTH_X = N_CORE_X * ASSEMBLY_PITCH
CORE_WIDTH_Y = N_CORE_Y * ASSEMBLY_PITCH
CORE_HALF_X = CORE_WIDTH_X / 2.0
CORE_HALF_Y = CORE_WIDTH_Y / 2.0

# -----------------------------------------------------------------------------
# Rectangular boundary of the 8x8 core lattice
# -----------------------------------------------------------------------------

xmin = openmoc.XPlane(x=-CORE_HALF_X, name="core lattice xmin")
xmax = openmoc.XPlane(x=+CORE_HALF_X, name="core lattice xmax")
ymin = openmoc.YPlane(y=-CORE_HALF_Y, name="core lattice ymin")
ymax = openmoc.YPlane(y=+CORE_HALF_Y, name="core lattice ymax")
zmin = openmoc.ZPlane(z=-ACTIVE_HEIGHT / 2.0, name="reactor zmin")
zmax = openmoc.ZPlane(z=+ACTIVE_HEIGHT / 2.0, name="reactor zmax")

# These are external boundaries
zmin.setBoundaryType(openmoc.VACUUM)
zmax.setBoundaryType(openmoc.VACUUM)

# -----------------------------------------------------------------------------
# Cylindrical radial regions
# -----------------------------------------------------------------------------
#
# R_MODERATOR_OUTER is chosen as 96.2 cm rather than exactly 96.0 cm
# so that the complete outer fuel assemblies are not clipped by the
# cylindrical boundary.
#
# Steel thicknesses inherited from the OpenMC (Maia) model:
#
#   SA508      : 5 cm
#   Steel 14404: 20 cm
#
# -----------------------------------------------------------------------------

R_MODERATOR_OUTER = 96.2
R_SA508_OUTER = R_MODERATOR_OUTER + 5.0
R_STEEL14404_OUTER = R_SA508_OUTER + 20.0

moderator_outer = openmoc.ZCylinder(x=0.0, y=0.0, radius=R_MODERATOR_OUTER, name="moderator outer radius")
sa508_outer = openmoc.ZCylinder(x=0.0, y=0.0, radius=R_SA508_OUTER, name="SA508 outer radius")
steel14404_outer = openmoc.ZCylinder(x=0.0, y=0.0, radius=R_STEEL14404_OUTER, name="Steel 14404 outer radius")

# Only the OUTERMOST cylinder is vacuum.
# moderator_outer and sa508_outer are internal interfaces and must
# remain transmissive (the OpenMOC default).

steel14404_outer.setBoundaryType(openmoc.VACUUM)

# SA508 ring
sa508_cell = openmoc.Cell(name="SA508 steel ring")
sa508_cell.setFill(Steel_SA508)
sa508_cell.addSurface(halfspace=+1, surface=moderator_outer)
sa508_cell.addSurface(halfspace=-1, surface=sa508_outer)
sa508_cell.addSurface(halfspace=+1, surface=zmin)
sa508_cell.addSurface(halfspace=-1, surface=zmax)

# Steel 1.4404 ring
steel14404_cell = openmoc.Cell(name="Steel 1.4404 outer ring")
steel14404_cell.setFill(Steel_14404)
steel14404_cell.addSurface(halfspace=+1, surface=sa508_outer)
steel14404_cell.addSurface(halfspace=-1, surface=steel14404_outer)
steel14404_cell.addSurface(halfspace=+1, surface=zmin)
steel14404_cell.addSurface(halfspace=-1, surface=zmax)

# -----------------------------------------------------------------------------
# Core lattice cell
# -----------------------------------------------------------------------------

core_cell = openmoc.Cell(name="BANDI-60 active-core cell")
core_cell.addSurface(halfspace=+1, surface=xmin)
core_cell.addSurface(halfspace=-1, surface=xmax)
core_cell.addSurface(halfspace=+1, surface=ymin)
core_cell.addSurface(halfspace=-1, surface=ymax)
core_cell.addSurface(halfspace=+1, surface=zmin)
core_cell.addSurface(halfspace=-1, surface=zmax)
core_cell.addSurface(halfspace=-1, surface=moderator_outer)
core_cell.setFill(core_lattice)


# =============================================================================
# WATER REGION AROUND THE 8x8 CORE LATTICE
# =============================================================================

# Top water region
water_top = openmoc.Cell(name="core peripheral water top")
water_top.setFill(water)
water_top.addSurface(halfspace=-1, surface=moderator_outer)
water_top.addSurface(halfspace=+1, surface=ymax)
water_top.addSurface(halfspace=+1, surface=zmin)
water_top.addSurface(halfspace=-1, surface=zmax)
# Bottom water region
water_bottom = openmoc.Cell(name="core peripheral water bottom")
water_bottom.setFill(water)
water_bottom.addSurface(halfspace=-1, surface=moderator_outer)
water_bottom.addSurface(halfspace=-1, surface=ymin)
water_bottom.addSurface(halfspace=+1, surface=zmin)
water_bottom.addSurface(halfspace=-1, surface=zmax)
# Left water region
water_left = openmoc.Cell(name="core peripheral water left")
water_left.setFill(water)
water_left.addSurface(halfspace=-1, surface=moderator_outer)
water_left.addSurface(halfspace=-1, surface=xmin)
water_left.addSurface(halfspace=+1, surface=ymin)
water_left.addSurface(halfspace=-1, surface=ymax)
water_left.addSurface(halfspace=+1, surface=zmin)
water_left.addSurface(halfspace=-1, surface=zmax)
# Right water region
water_right = openmoc.Cell(name="core peripheral water right")
water_right.setFill(water)
water_right.addSurface(halfspace=-1, surface=moderator_outer)
water_right.addSurface(halfspace=+1, surface=xmax)
water_right.addSurface(halfspace=+1, surface=ymin)
water_right.addSurface(halfspace=-1, surface=ymax)
water_right.addSurface(halfspace=+1, surface=zmin)
water_right.addSurface(halfspace=-1, surface=zmax)

# =============================================================================
# ROOT UNIVERSE
# =============================================================================

root_universe = openmoc.Universe(name="BANDI-60 root universe")
root_universe.addCell(core_cell)
root_universe.addCell(water_top)
root_universe.addCell(water_bottom)
root_universe.addCell(water_right)
root_universe.addCell(water_left)
root_universe.addCell(sa508_cell)
root_universe.addCell(steel14404_cell)


# =============================================================================
# CREATE OPENMOC GEOMETRY
# =============================================================================

log.py_printf("NORMAL", "Creating BANDI-60 OpenMOC geometry...")

geometry = openmoc.Geometry()

geometry.setRootUniverse(root_universe)

geometry.initializeFlatSourceRegions()

def check_material_at(x, y):

    material_dict = geometry.getAllMaterials()

    material_id = geometry.getSpatialDataOnGrid(
        np.array([x]),
        np.array([y]),
        offset=0.0,
        plane='xy',
        domain_type='material'
    )[0]

    if material_id in material_dict:
        name = material_dict[material_id].getName()
    else:
        name = "NO MATERIAL / outside geometry"

    print(
        "x = {}, y = {} -> ID {} -> {}".format(
            x, y, material_id, name
        )
    )

check_material_at(0.0, 0.0)
check_material_at(90.0, 0.0)
check_material_at(0.0, 90.0)
check_material_at(95.0, 0.0)
check_material_at(98.0, 0.0)
check_material_at(110.0, 0.0)
check_material_at(125.0, 0.0)


# =============================================================================
# SANITY INFORMATION
# =============================================================================

log.py_printf(
    "NORMAL",
    "BANDI-60 geometry: 52 FAs; FA counts = "
    "5%%:%d, 10%%:%d, 25%%:%d, 35%%:%d, 40%%:%d",
    assembly_counts[5],
    assembly_counts[10],
    assembly_counts[25],
    assembly_counts[35],
    assembly_counts[40]
)

log.py_printf(
    "NORMAL",
    "Each FA: 240 fuel pins, 24 Pyrex pins, "
    "24 guide tubes, 1 instrument tube"
)

log.py_printf(
    "NORMAL",
    "Pin pitch = %.3f cm; FA pitch = %.3f cm; "
    "active height = %.1f cm",
    PIN_PITCH,
    ASSEMBLY_PITCH,
    ACTIVE_HEIGHT
)


# ============================================================
# PLOT
# ============================================================

# Directory containing geometry.py
BASE_DIR = Path(__file__).resolve().parent

# BANDI60/plots
PLOTS_DIR = BASE_DIR / "plots"

MATERIAL_COLORS = {

    "UO2": "#F31CD6",          # pink
    "water": "#0066CC",        # blue
    "pyrex40": "#E31A1C",      # red
    "pyrex35": "#FF4500",      # red-orange
    "pyrex25": "#FF8C00",      # orange
    "pyrex10": "#FFC107",      # orange-yellow
    "pyrex5": "#FFFF00",       # yellow
    "zircaloy4": "#ADD8E6",    # light blue
    "Steel_14404": "#D3D3D3",  # light gray
    "Steel_SA508": "#555555",   # dark gray
    "helium": "#90EE90",       # light green
}

def plot_materials_custom(geometry, gridsize=1000, plane="xy", offset=0.0, xlim=None, ylim=None, zlim=None):

    material_dict = geometry.getAllMaterials()

    # Material name -> OpenMOC material ID
    name_to_id = {
        material.getName(): material_id
        for material_id, material in material_dict.items()
    }

    # Make sure every material is included in our plotting definition
    unknown_materials = [
        name
        for name in name_to_id
        if name not in LEGEND_ORDER
    ]

    if unknown_materials:
        raise ValueError(
            "Materials missing from LEGEND_ORDER: {}".format(
                unknown_materials
            )
        )

    # ----------------------------------------------------------
    # IMPORTANT:
    # value 0 is reserved for points outside the geometry.
    # Real materials therefore start from 1.
    # ----------------------------------------------------------

    material_to_plot_value = {}

    # First colour = background
    plot_colors = ["white"]

    for name in LEGEND_ORDER:

        if name in name_to_id:
            plot_value = len(plot_colors)
            material_to_plot_value[name_to_id[name]] = plot_value
            plot_colors.append(MATERIAL_COLORS[name])

    # Custom deterministic colour map
    cmap = ListedColormap(plot_colors)

    # OpenMOC plotting parameters
    params = plotter.PlotParams()
    params.geometry = geometry
    params.domain_type = "material"
    params.gridsize = gridsize
    params.plane = plane
    params.offset = offset
    params.xlim = xlim
    params.ylim = ylim
    params.zlim = zlim
    params.interpolation = "nearest"
    params.cmap = cmap
    params.vmin = 0
    params.vmax = len(plot_colors) - 1
    params.suptitle = "Materials"

    if plane == "xy":
        params.title = "z = {}".format(offset)
        params.filename = "custom-materials-z-{}".format(offset)
    elif plane == "xz":
        params.title = "y = {}".format(offset)
        params.filename = "custom-materials-y-{}".format(offset)
    elif plane == "yz":
        params.title = "x = {}".format(offset)
        params.filename = "custom-materials-x-{}".format(offset)

    figures = plotter.plot_spatial_data(material_to_plot_value, params, get_figure=True)
    fig = figures[0]
    fig.patch.set_facecolor("white")
    ax = fig.axes[0]
    ax.set_facecolor("white")

    return fig

# Include materials color legend
LEGEND_ORDER = ["UO2", "water", "helium", "zircaloy4", "pyrex5", "pyrex10", "pyrex25", "pyrex35", "pyrex40", "Steel_SA508", "Steel_14404"]
MATERIAL_LABELS = {
    "UO2": "UO$_2$", "water": "Water", "helium": "Helium", "zircaloy4": "Zircaloy-4", "pyrex5": "Pyrex 5%", "pyrex10": "Pyrex 10%",
    "pyrex25": "Pyrex 25%", "pyrex35": "Pyrex 35%", "pyrex40": "Pyrex 40%", "Steel_SA508": "Steel SA508", "Steel_14404": "Steel 1.4404",
}

def add_material_legend(fig, geometry):

    ax = fig.axes[0]

    # Materials that actually exist in this geometry
    material_dict = geometry.getAllMaterials()
    available_names = {
        material.getName()
        for material in material_dict.values()
    }
    legend_handles = []

    for name in LEGEND_ORDER:

        if name in available_names:
            legend_handles.append(
                Patch(
                    facecolor=MATERIAL_COLORS[name],
                    edgecolor="black",
                    label=MATERIAL_LABELS[name]
                )
            )

    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=8
    )

# CORE XY

fig = plot_materials_custom(geometry,  gridsize=3500, plane='xy', offset=0.0, xlim=(-130, 130), ylim=(-130, 130))
add_material_legend(fig, geometry)
fig.savefig(PLOTS_DIR /"core_xy.pdf", bbox_inches="tight")
plt.close(fig)

# PIN / ASSEMBLY ZOOM

fig = plot_materials_custom(geometry,  gridsize=3500, plane='xy', offset=0.0, xlim=(0, 21.5), ylim=(0, 21.5))
add_material_legend(fig, geometry)
fig.savefig(PLOTS_DIR /"pin_pyrex40_xy.pdf", bbox_inches="tight")
plt.close(fig)

fig = plot_materials_custom(geometry,  gridsize=3500, plane='xy', offset=0.0, xlim=(21.5, 43), ylim=(0, 21.5))
add_material_legend(fig, geometry)
fig.savefig(PLOTS_DIR /"pin_pyrex35_xy.pdf", bbox_inches="tight")
plt.close(fig)

fig = plot_materials_custom(geometry,  gridsize=3500, plane='xy', offset=0.0, xlim=(43, 64.5), ylim=(0, 21.5))
add_material_legend(fig, geometry)
fig.savefig(PLOTS_DIR /"pin_pyrex25_xy.pdf", bbox_inches="tight")
plt.close(fig)

fig = plot_materials_custom(geometry,  gridsize=3500, plane='xy', offset=0.0, xlim=(64.5, 86), ylim=(0, 21.5))
add_material_legend(fig, geometry)
fig.savefig(PLOTS_DIR /"pin_pyrex10_xy.pdf", bbox_inches="tight")
plt.close(fig)

fig = plot_materials_custom(geometry,  gridsize=3500, plane='xy', offset=0.0, xlim=(64.5, 86), ylim=(21.5, 43))
add_material_legend(fig, geometry)
fig.savefig(PLOTS_DIR /"pin_pyrex5_xy.pdf", bbox_inches="tight")
plt.close(fig)

# CORE XZ

fig = plot_materials_custom(geometry,   gridsize=3500, plane='xz', offset=50.0, xlim=(-100, 100), zlim=(-100, 100))
add_material_legend(fig, geometry)
fig.savefig(PLOTS_DIR /"core_xz.pdf",  bbox_inches="tight")
plt.close(fig)