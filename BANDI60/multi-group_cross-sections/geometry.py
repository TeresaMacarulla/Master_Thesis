import openmc
import openmc.deplete
import numpy as np
import matplotlib.pyplot as plt
from materials import materials
import pandas as pd

# Convenient material lookup by name
mat = {material.name: material for material in materials}

PIN_PITCH = 1.26
ASSEMBLY_PITCH = 21.50
ACTIVE_HEIGHT = 200.0

N_PIN = 17

PIN_LATTICE_WIDTH = N_PIN * PIN_PITCH   # 21.42 cm
ASSEMBLY_HALF_GAP = (
    ASSEMBLY_PITCH - PIN_LATTICE_WIDTH
) / 2.0                                  # 0.04 cm

pitch = PIN_PITCH
height = 2.0

" ################ PINCELL MODEL ################## "
#pincell radii [cm]
r0 = 0.4096
r1 = 0.4180
r2 = 0.4750

#fuel pellet
pitch = 1.26 #cm
height = 2.0 #cm ---> z-dir = 100

#fuel/gap/clad regions
fuel_cyl = openmc.ZCylinder(x0=0.0, y0=0.0, r=r0)
gap_cyl = openmc.ZCylinder(x0=0.0, y0=0.0, r=r1)
clad_cyl = openmc.ZCylinder(x0=0.0, y0=0.0, r=r2)

#boundaries
ymax = openmc.YPlane(y0 = pitch/2.0)
ymin = openmc.YPlane(y0 = -pitch/2.0)
xmin = openmc.XPlane(x0 = -pitch/2.0)
xmax = openmc.XPlane(x0 = pitch/2.0)
zmax = openmc.ZPlane(z0 = height/2.0)
zmin = openmc.ZPlane(z0 = -height/2.0)

#regions and surfaces
fuel_reg = -fuel_cyl
gap_reg = +fuel_cyl & -gap_cyl
clad_reg = +gap_cyl & -clad_cyl
pyr_reg = fuel_reg

for surf in [ymax, ymin, xmin, xmax, zmax, zmin]:
    surf.boundary_type="transmission"

mod_reg = +clad_cyl & -ymax & +ymin & +xmin & -xmax & -zmax & +zmin

#cell assignment - fuel pin
fuel_cell = openmc.Cell(region=fuel_reg, fill=mat["UO2"])
gap_cell = openmc.Cell(region=gap_reg, fill=mat["helium"])
clad_cell = openmc.Cell(region=clad_reg, fill=mat["zircaloy4"])
mod_cell = openmc.Cell(region=mod_reg, fill=mat["water"])

#cell assignment - pyrex pins
pyr_cells = [openmc.Cell(region=pyr_reg, fill=materials[i]) for i in range(4, 9)]
py_gap_cells = [openmc.Cell(region=gap_reg, fill=mat["helium"]) for _ in range(5)]
py_clad_cells = [openmc.Cell(region=clad_reg, fill=mat["zircaloy4"]) for _ in range(5)]
py_mod_cells = [openmc.Cell(region=mod_reg, fill=mat["water"]) for _ in range(5)]

#universe creation
f_universe = openmc.Universe(name="fuelpin_universe", cells=[fuel_cell, gap_cell, clad_cell, mod_cell])
py_universe = [openmc.Universe(name=f"Pyrex{i}_universe", cells=[pyr_cells[i], py_gap_cells[i], py_clad_cells[i], py_mod_cells[i]])
    for i in range(5)]

R_GUIDE_INNER = 0.561
R_GUIDE_OUTER = 0.602

def make_tube_universe(name):

    inner = openmc.ZCylinder(r=R_GUIDE_INNER)
    outer = openmc.ZCylinder(r=R_GUIDE_OUTER)

    inner_water = openmc.Cell(name=f"{name} inner water", region=-inner, fill=mat["water"])
    tube_wall = openmc.Cell(name=f"{name} Zircaloy wall", region=+inner & -outer, fill=mat["zircaloy4"])
    outer_water = openmc.Cell(name=f"{name} outer water", region=+outer, fill=mat["water"])

    return openmc.Universe(name=f"{name} universe",
        cells=[inner_water, tube_wall, outer_water])

guide_universe = make_tube_universe("guide tube")
instrument_universe = make_tube_universe("instrument tube")

" ################# END PIN CELL MODEL ######################### "


" ################ ASSEMBLY MODEL ############################## "
#define fuel assembly lattice size- and boundaries
x_assembly_lattice = 17
y_assembly_lattice = 17
z_assembly_lattice = 100

min_x_assembly = openmc.XPlane(x0=-ASSEMBLY_PITCH / 2.0)
max_x_assembly = openmc.XPlane(x0=+ASSEMBLY_PITCH / 2.0)
min_y_assembly = openmc.YPlane(y0=-ASSEMBLY_PITCH / 2.0)
max_y_assembly = openmc.YPlane(y0=+ASSEMBLY_PITCH / 2.0)
min_z_assembly = openmc.ZPlane(z0=-ACTIVE_HEIGHT / 2.0)
max_z_assembly = openmc.ZPlane(z0=+ACTIVE_HEIGHT / 2.0)

for surf in [min_x_assembly, max_x_assembly, min_y_assembly, max_y_assembly, min_z_assembly, max_z_assembly]:
    surf.boundary_type = "transmission"

#surround assembly w moderator
assembly_mod_region = +min_x_assembly & -max_x_assembly & +min_y_assembly & -max_y_assembly & +min_z_assembly & -max_z_assembly
assembly_mod_cell = openmc.Cell(name="assembly outer moderator cell", region=assembly_mod_region, fill=mat["water"])
assembly_mod_universe = openmc.Universe(name="assembly moderator universe", cells=[assembly_mod_cell])

#create fuel assembly lattice
def create_assembly(n):
    #assert 0 <= n <=5, "Nope not valid universe index!"
    assembly_lat = openmc.RectLattice(lattice_id=1000+n,name=f"FA_lattice_{n}")
    assembly_lat.pitch = (PIN_PITCH, PIN_PITCH, height)
    assembly_lat.lower_left = [-PIN_LATTICE_WIDTH / 2.0, -PIN_LATTICE_WIDTH / 2.0, -ACTIVE_HEIGHT / 2.0]
    #fill w fuel universes
    assembly_lat.universes = [[[f_universe]*x_assembly_lattice]*y_assembly_lattice]*z_assembly_lattice
    assembly_lat.outer = assembly_mod_universe

    assembly_region = (+min_x_assembly & -max_x_assembly & +min_y_assembly & -max_y_assembly & +min_z_assembly & -max_z_assembly)
    assembly_lat_cell = openmc.Cell(name=f"assembly_{n}", region=assembly_region, fill=assembly_lat)

    #define specific positions for pyrex rods and void universes
    pyrex_positions = [(1,5),(1,11),(2,2),(2,14),(3,8),(4,4),(4,12),(5,1),(5,15),(6,6),(6,10),(8,1),(8,15)]
    void_positions = [(2,5),(2,8),(2,11),(3,3),(3,13),(5,2),(5,5),(5,8),(5,11),(5,14),(8,2),(8,5),(8,8),(8,11),(8,14)]

    #upper half assembly
    for y in range(9):
        for x in range(16):
            for z in range(z_assembly_lattice):
                if (y, x) in pyrex_positions:
                    assembly_lat.universes[z][y][x] = py_universe[n]
                elif (y, x) in void_positions:
                    assembly_lat.universes[z][y][x] = guide_universe
                else:
                    pass


    #bottom half assembly
    for y in range(9):
        for x in range(16):
            for z in range(z_assembly_lattice):
                    assembly_lat.universes[z][16-y][16-x] = assembly_lat.universes[z][y][x]

    assembly_lat_universe = openmc.Universe(name=f"assembly_universe_{n}", cells=[assembly_lat_cell])

    return assembly_lat_universe

assemblies = [create_assembly(i) for i in range(5)]

"############## END ASSEMBLY MODEL ###########################"

"############# BEGIN CORE MODEL ################################"

#define core param
x_core_lattice = 8
y_core_lattice = 8
z_core_lattice = 1

core_mod_cell = openmc.Cell(name="core outer moderator cell", fill=mat["water"])
core_mod_universe = openmc.Universe(name="core outer moderator universe", cells=[core_mod_cell])

core_lat = openmc.RectLattice(lattice_id=1100, name="core_lattice")
CORE_HALF = 4.0 * ASSEMBLY_PITCH   # 86.0 cm
core_lat.lower_left = [-CORE_HALF, -CORE_HALF, -ACTIVE_HEIGHT / 2.0]
core_lat.pitch = (ASSEMBLY_PITCH, ASSEMBLY_PITCH, ACTIVE_HEIGHT)
core_lat.universes = [[[assembly_mod_universe]*x_core_lattice]*y_core_lattice]*z_core_lattice
core_lat.outer = core_mod_universe

#create loop to fill core w respective assemblies
py_5_positions = [(0,2),(0,5),(2,0),(2,7),(5,0),(5,7),(7,2),(7,5)]
py_10_positions = [(0,3),(0,4),(1,1),(1,6),(3,0),(3,7),(4,0),(4,7),(6,1),(6,6),(7,3),(7,4)]
py_25_positions = [(1,2),(1,3),(1,4),(1,5),(2,1),(2,6),(3,1),(3,6),(4,1),(4,6),(5,1),(5,6),
                  (6,2),(6,3),(6,4),(6,5)]
py_35_positions = [(2,2),(2,3),(2,4),(2,5),(3,2),(3,5),(4,2),(4,5),(5,2),(5,3),(5,4),(5,5)]
py_40_positions = [(3,3),(3,4),(4,3),(4,4)]


for y in range(8):
    for x in range(8):
        for z in range(z_core_lattice):
            if (y, x) in py_5_positions:
                 core_lat.universes[z][y][x] = assemblies[0]
            elif (y, x) in py_10_positions:
                core_lat.universes[z][y][x] = assemblies[1]
            elif (y, x) in py_25_positions:
                core_lat.universes[z][y][x] = assemblies[2]
            elif (y, x) in py_35_positions:
                core_lat.universes[z][y][x] = assemblies[3]
            elif (y, x) in py_40_positions:
                core_lat.universes[z][y][x] = assemblies[4]
            else:
                pass


"##################### END CORE MODEL ########################"

"################## SHIELDING ############################"
#TERESA; I did not spend a lot of time looking for shielding specification for the BANDI
#I think I just made some educated guesses given what I had of information and what I knew from courses
#Don't know how important that is for ur use just thought I should mention it

R_MODERATOR_OUTER = 96.2
R_SA508_OUTER = 101.2
R_STEEL14404_OUTER = 121.2

ACTIVE_Z_HALF = ACTIVE_HEIGHT / 2.0
AXIAL_REFLECTOR_THICKNESS = 20.0

DOMAIN_HALF = R_STEEL14404_OUTER + 1.0
DOMAIN_Z_HALF = (ACTIVE_Z_HALF + AXIAL_REFLECTOR_THICKNESS)

#core-, steel- and shielding cylinders
moderator_outer = openmc.ZCylinder(r=R_MODERATOR_OUTER)
sa508_outer = openmc.ZCylinder( r=R_SA508_OUTER)
steel14404_outer = openmc.ZCylinder(r=R_STEEL14404_OUTER)
domain_xmin = openmc.XPlane(x0=-DOMAIN_HALF, boundary_type="vacuum")
domain_xmax = openmc.XPlane(x0=+DOMAIN_HALF, boundary_type="vacuum")
domain_ymin = openmc.YPlane(y0=-DOMAIN_HALF, boundary_type="vacuum")
domain_ymax = openmc.YPlane(y0=+DOMAIN_HALF, boundary_type="vacuum")
domain_zmin = openmc.ZPlane(z0=-DOMAIN_Z_HALF, boundary_type="vacuum")
domain_zmax = openmc.ZPlane(z0=+DOMAIN_Z_HALF, boundary_type="vacuum")

outer_buffer_region = (+steel14404_outer & +domain_xmin & -domain_xmax & +domain_ymin & -domain_ymax & +domain_zmin & -domain_zmax)
outer_buffer = openmc.Cell(name="outside-vessel buffer", fill=None, region=outer_buffer_region)

ACTIVE_Z_HALF = ACTIVE_HEIGHT / 2.0

# Provisional sensitivity-study value.
# NOT obtained from Kim.
AXIAL_REFLECTOR_THICKNESS = 20.0

DOMAIN_Z_HALF = (ACTIVE_Z_HALF + AXIAL_REFLECTOR_THICKNESS)
active_zmin = openmc.ZPlane(z0=-ACTIVE_Z_HALF)
active_zmax = openmc.ZPlane(z0=+ACTIVE_Z_HALF)
domain_zmin = openmc.ZPlane(z0=-DOMAIN_Z_HALF, boundary_type="vacuum")
domain_zmax = openmc.ZPlane(z0=+DOMAIN_Z_HALF, boundary_type="vacuum")

core_region = (-moderator_outer & +active_zmin & -active_zmax)
core_cell = openmc.Cell(name="active core", region=core_region, fill=core_lat)
top_reflector_region = (-moderator_outer & +active_zmax & -domain_zmax)
bottom_reflector_region = (-moderator_outer & +domain_zmin & -active_zmin)
top_reflector = openmc.Cell(name="upper axial reflector", region=top_reflector_region, fill=mat["water"])
bottom_reflector = openmc.Cell(name="lower axial reflector", region=bottom_reflector_region, fill=mat["water"])

sa508_region = (+moderator_outer & -sa508_outer & +domain_zmin & -domain_zmax)
steel14404_region = ( +sa508_outer & -steel14404_outer & +domain_zmin & -domain_zmax)
sa508_cell = openmc.Cell(name="sa508 cell", region=sa508_region, fill=mat["Steel_SA508"])
steel14404_cell = openmc.Cell(name="steel14404 cell", region=steel14404_region, fill=mat["Steel_14404"])

reactor_universe = openmc.Universe(
    name="final universe",
    cells=[core_cell, top_reflector, bottom_reflector, sa508_cell, steel14404_cell, outer_buffer]
)

geometry = openmc.Geometry(reactor_universe)
geometry.export_to_xml()


"################# SIMULATION SETUP AND MODEL CREATION ######################"
#define source
source = openmc.IndependentSource()
source.space = openmc.stats.Box(
    [-CORE_HALF, -CORE_HALF, -ACTIVE_Z_HALF],
    [+CORE_HALF, +CORE_HALF, +ACTIVE_Z_HALF]
)
source.constraints = {"fissionable": True}
source.energy = openmc.stats.Watt(a=0.988e6, b=2.249e-6) 

# SETTINGS USED FOR FIRST DEPLETION BENCHMARK
settings = openmc.Settings()
settings.source = source
settings.particles = 100000
settings.generations_per_batch = 1
settings.batches = 450
settings.inactive = 150
settings.run_mode = "eigenvalue"
settings.temperature = {"method": "interpolation"}

settings.export_to_xml()







