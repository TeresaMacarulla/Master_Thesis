"""
geometry_pin.py
===========

OpenMOC geometry for the fuel pin of the BANDI-60 active core.
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
MGXS_DIRECTORY = "../OpenMC_model/"

log.set_log_level("NORMAL")

# =============================================================================
# GEOMETRIC DATA [cm]
# =============================================================================

# Fuel / Pyrex pin radii
R_FUEL = 0.4096
R_GAP = 0.4180
R_CLAD = 0.4750

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
    fuel.setNumRings(2)
    fuel.setNumSectors(4)

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
    moderator.setNumRings(2)
    moderator.setNumSectors(4)

    universe = openmoc.Universe(name="fuel pin universe")
    universe.addCell(fuel)
    universe.addCell(gap)
    universe.addCell(clad)
    universe.addCell(moderator)

    return universe

fuel_pin = make_fuel_pin_universe()

# =============================================================================
# CREATE OPENMOC GEOMETRY
# =============================================================================

log.py_printf("NORMAL", "Creating BANDI-60 fuel pin OpenMOC geometry...")

geometry = openmoc.Geometry()
geometry.setRootUniverse(fuel_pin)
geometry.initializeFlatSourceRegions()




