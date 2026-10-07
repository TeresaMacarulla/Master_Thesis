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
    inner_water.setFill(water)
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
# ROOT ACTIVE-CORE BOUNDARIES
# =============================================================================

CORE_WIDTH_X = N_CORE_X * ASSEMBLY_PITCH
CORE_WIDTH_Y = N_CORE_Y * ASSEMBLY_PITCH

xmin = openmoc.XPlane(x=-CORE_WIDTH_X / 2.0, name="active-core xmin")
xmax = openmoc.XPlane(x=+CORE_WIDTH_X / 2.0, name="active-core xmax")
ymin = openmoc.YPlane(y=-CORE_WIDTH_Y / 2.0, name="active-core ymin")
ymax = openmoc.YPlane(y=+CORE_WIDTH_Y / 2.0, name="active-core ymax")
zmin = openmoc.ZPlane(z=-ACTIVE_HEIGHT / 2.0, name="active-core zmin")
zmax = openmoc.ZPlane(z=+ACTIVE_HEIGHT / 2.0, name="active-core zmax")

# Vacuum is applied at the boundary of this active-core-only model.
for surface in (xmin, xmax, ymin, ymax, zmin, zmax):
    surface.setBoundaryType(openmoc.VACUUM)


root_cell = openmoc.Cell(name="BANDI-60 active-core cell")
root_cell.addSurface(halfspace=+1, surface=xmin)
root_cell.addSurface(halfspace=-1, surface=xmax)
root_cell.addSurface(halfspace=+1, surface=ymin)
root_cell.addSurface(halfspace=-1, surface=ymax)
root_cell.addSurface(halfspace=+1, surface=zmin)
root_cell.addSurface(halfspace=-1, surface=zmax)
root_cell.setFill(core_lattice)

root_universe = openmoc.Universe(name="BANDI-60 root universe")
root_universe.addCell(root_cell)


# =============================================================================
# CREATE OPENMOC GEOMETRY
# =============================================================================

log.py_printf("NORMAL", "Creating BANDI-60 OpenMOC geometry...")

geometry = openmoc.Geometry()

geometry.setRootUniverse(root_universe)

geometry.initializeFlatSourceRegions()


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

plotter.plot_materials(
    geometry,
    gridsize=1000,
    plane='xy',
    offset=0.0,
    xlim=(-86, 86),
    ylim=(-86, 86)
)

plotter.plot_materials(
    geometry,
    gridsize=1000,
    plane='xz',
    offset=0.0,
    xlim=(-86, 86),
    zlim=(-100, 100)
)
