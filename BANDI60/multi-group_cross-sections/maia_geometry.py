import openmc
import openmc.deplete
import numpy as np
import matplotlib.pyplot as plt
from maia_materials import materials
import pandas as pd
#import seaborn as sns


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
fuel_cell = openmc.Cell(region=fuel_reg, fill=materials[0])
gap_cell = openmc.Cell(region=gap_reg, fill=materials[1])
clad_cell = openmc.Cell(region=clad_reg, fill=materials[2])
mod_cell = openmc.Cell(region=mod_reg, fill=materials[3])

#cell assignment - pyrex pins
pyr_cells = [openmc.Cell(region=pyr_reg, fill=materials[i]) for i in range(4, 9)]
py_gap_cells = [openmc.Cell(region=gap_reg, fill=materials[1]) for _ in range(5)]
py_clad_cells = [openmc.Cell(region=clad_reg, fill=materials[2]) for _ in range(5)]
py_mod_cells = [openmc.Cell(region=mod_reg, fill=materials[3]) for _ in range(5)]

#cell assignment - instrumentation/guidetubes
void_cell = openmc.Cell(region=fuel_reg, fill=materials[1])
void_gap_cell = openmc.Cell(region=gap_reg, fill=materials[1])
void_clad_cell = openmc.Cell(region=clad_reg, fill=materials[1])
void_mod_cell = openmc.Cell(region=mod_reg, fill=materials[3])

#universe creation
f_universe = openmc.Universe(name="fuelpin_universe", cells=[fuel_cell, gap_cell, clad_cell, mod_cell])
py_universe = [openmc.Universe(name=f"Pyrex{i}_universe", cells=[pyr_cells[i], py_gap_cells[i], py_clad_cells[i], py_mod_cells[i]])
    for i in range(5)]

void_universe = openmc.Universe(name="void_pin_universe", cells=[void_cell, void_gap_cell, void_clad_cell, void_mod_cell])

"""
"PLOT PINCELL"
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("talk")  # larger fonts for thesis/presentations


material_colors1 = {
    materials[4]: "lightpink",
    materials[1]: "khaki",
    materials[2]: "darkkhaki",
    materials[3]: "lightblue",
    # You can add more materials if needed
}


plot_args = {
    "width" : (1, 1),
    "pixels" : (800, 800),
    "color_by" : "material",
    "colors" : material_colors1,
    "basis" : "xy",
}


py_universe[0].plot(**plot_args, legend=True)

plt.xlabel("x [cm]", fontsize=14)
plt.ylabel("y [cm]", fontsize=14)
plt.title("Cross-section of Burnable Absorber pin (xy-plane)", fontsize=16, weight="bold")


plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig("py_pin_xy_plot_uo2.png", dpi=300)
plt.show()
"""



" ################# END PIN CELL MODEL ######################### "


" ################ ASSEMBLY MODEL ############################## "
#define fuel assembly lattice size- and boundaries
x_assembly_lattice = 17
y_assembly_lattice = 17
z_assembly_lattice = 100

min_x_assembly = openmc.XPlane(x0=-pitch*x_assembly_lattice/2.0)
max_x_assembly = openmc.XPlane(x0=+pitch*x_assembly_lattice/2.0)
min_y_assembly = openmc.YPlane(y0=-pitch*y_assembly_lattice/2.0)
max_y_assembly = openmc.YPlane(y0=+pitch*y_assembly_lattice/2.0)
min_z_assembly = openmc.ZPlane(z0=-z_assembly_lattice)
max_z_assembly = openmc.ZPlane(z0=+z_assembly_lattice)

for surf in [min_x_assembly, max_x_assembly, min_y_assembly, max_y_assembly, min_z_assembly, max_z_assembly]:
    surf.boundary_type = "transmission"

#surround assembly w moderator
assembly_mod_region = +min_x_assembly & -max_x_assembly & +min_y_assembly & -max_y_assembly & +min_z_assembly & -max_z_assembly
assembly_mod_cell = openmc.Cell(name="assembly outer moderator cell", region=assembly_mod_region, fill=materials[3])
assembly_mod_universe = openmc.Universe(name="assembly moderator universe", cells=[assembly_mod_cell])

#create fuel assembly lattice
def create_assembly(n):
    #assert 0 <= n <=5, "Nope not valid universe index!"
    assembly_lat = openmc.RectLattice(lattice_id=10+(n+3), name=f"FA_lattice_{n}")
    assembly_lat.pitch = (pitch, pitch, height)
    assembly_lat.lower_left = [-pitch*x_assembly_lattice/2.0, -pitch*y_assembly_lattice/2.0, -z_assembly_lattice]
    #fill w fuel universes
    assembly_lat.universes = [[[f_universe]*x_assembly_lattice]*y_assembly_lattice]*z_assembly_lattice
    assembly_lat.outer = assembly_mod_universe

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
                    assembly_lat.universes[z][y][x] = void_universe
                else:
                    pass


    #bottom half assembly
    for y in range(9):
        for x in range(16):
            for z in range(z_assembly_lattice):
                    assembly_lat.universes[z][16-y][16-x] = assembly_lat.universes[z][y][x]

    assembly_lat_cell = openmc.Cell(name=f"assembly_{n}", fill=assembly_lat)
    assembly_lat_universe = openmc.Universe(name=f"assembly_universe_{n}", cells=[assembly_lat_cell])

    return assembly_lat_universe

assemblies = [create_assembly(i) for i in range(5)]

"""
"PLOT ASSEMBLY"
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("talk")

material_colors2 = {
    materials[0]: "lemonchiffon",
    materials[1]: "khaki",
    materials[2]: "darkkhaki",
    materials[3]: "lightblue",
    materials[4]: "lightpink",
    #materials[5]: "lightcoral",
    #materials[6]: "indianred",
    #materials[7]: "firebrick",
    #materials[8]: "darkred"

}


plot_args2 = {
    "width" : (23,23),
    "pixels" : (900,900),
    "color_by" : "material",
    "colors" : material_colors2,
    "basis" : "xy",
}


assemblies[0].plot(**plot_args2, legend=True)
plt.xlabel("x [cm]", fontsize=14)
plt.ylabel("y [cm]", fontsize=14)
plt.title("Cross-section of Fuel Assembly (xy-plane)", fontsize=16, weight="bold")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.gca().set_aspect('equal', adjustable='box')  # Keep aspect ratio square
plt.tight_layout()
plt.savefig("assembly_xy_plot_new.png", dpi=300)
plt.show()
"""



"############## END ASSEMBLY MODEL ###########################"

"############# BEGIN CORE MODEL ################################"
# volume of one pin-cell: np.pi*clad_cyl**2*height
# number of pincells on top of each other (z-dir): z_fuel_lattice
# number of pin cells in fuel assembly: (x_fuel_lattice*y_fuel_lattice-49)
# number of fuel assemblies in core: 52
#UO2.volume = np.pi*clad_cyl**2*height*z_fuel_lattice*(x_fuel_lattice*y_fuel_lattice-49)*52

#define core param
x_core_lattice = 8
y_core_lattice = 8
z_core_lattice = 1

min_x_core = openmc.XPlane(x0=-pitch*x_assembly_lattice*x_core_lattice/2)
max_x_core = openmc.XPlane(x0=+pitch*x_assembly_lattice*x_core_lattice/2)
min_y_core = openmc.YPlane(y0=-pitch*y_assembly_lattice*y_core_lattice/2)
max_y_core = openmc.YPlane(y0=+pitch*y_assembly_lattice*y_core_lattice/2)
min_z_core = openmc.ZPlane(z0=-z_assembly_lattice*z_core_lattice)
max_z_core = openmc.ZPlane(z0=+z_assembly_lattice*z_core_lattice)

for surf in [min_x_core, max_x_core, min_y_core, max_y_core, min_z_core, max_z_core]:
    surf.boundary_type = "vacuum"

#surround core w moderator
core_mod_region = +min_x_core & -max_x_core & +min_y_core & -max_y_core & +min_z_core & -max_z_core
core_mod_cell = openmc.Cell(name="core moderator cell", region=core_mod_region, fill=materials[3])
core_mod_universe = openmc.Universe(name="core moderator universe", cells=[core_mod_cell])

core_lat = openmc.RectLattice(lattice_id=50, name="core_lattice")
core_lat.lower_left = [-pitch*x_assembly_lattice*x_core_lattice/2.0, -pitch*y_assembly_lattice*y_core_lattice/2.0, -z_assembly_lattice]
core_lat.pitch = (pitch*x_assembly_lattice, pitch*y_assembly_lattice, height*z_assembly_lattice) #Distance between centers of fuel assembly - why did you add 0.08?
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


#core_lat_cell = openmc.Cell(name="core lattice cell", fill=core_lat)
#core_lat_universe = openmc.Universe(name="root universe", cells=[core_lat_cell])

#print(core_lat)
#core_lat_universe.plot(**plot_args3)
#plt.show()

"##################### END CORE MODEL ########################"

"################## SHIELDING ############################"
#TERESA; I did not spend a lot of time looking for shielding specification for the BANDI
#I think I just made some educated guesses given what I had of information and what I knew from courses
#Don't know how important that is for ur use just thought I should mention it

#core-, steel- and shielding cylinders
core_outer = openmc.ZCylinder(r=(pitch*x_assembly_lattice*x_core_lattice/2.0)+10, boundary_type="transmission")
steel_outer = openmc.ZCylinder(r=(pitch*x_assembly_lattice*x_core_lattice/2.0)+15, boundary_type="transmission")
shielding_outer = openmc.ZCylinder(r=(pitch*x_assembly_lattice*x_core_lattice/2.0)+35, boundary_type="vacuum")

#define core-, steel and shielding regions
core_region = -core_outer & +min_z_core & -max_z_core
steel_region = +core_outer & -steel_outer & +min_z_core & -max_z_core
shielding_region = +steel_outer & -shielding_outer & +min_z_core & -max_z_core

#define cells for core-, steel and shielding
core_cell = openmc.Cell(name="core lattice cell", region=core_region, fill=core_lat)
steel_cell = openmc.Cell(name="steel cell", region=steel_region, fill=materials[9])
shielding_cell = openmc.Cell(name="shielding cell", region=shielding_region, fill=materials[10])

reactor_universe = openmc.Universe(name="final universe", cells=[core_cell, steel_cell, shielding_cell])

"""
"PLOT FINAL CORE"

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("talk")

material_colors3 = {
    materials[0]: "lemonchiffon",
    materials[1]: "khaki",
    materials[2]: "darkkhaki",
    materials[3]: "lightblue",
    materials[4]: "lightpink",
    materials[5]: "lightcoral",
    materials[6]: "indianred",
    materials[7]: "firebrick",
    materials[8]: "darkred",
    materials[9]: "grey",
    materials[10]: "gainsboro"

}

plot_args3 = {
    "width" : (225, 225),
    "pixels" : (1000,1000),
    "color_by" : "material",
    "colors" : material_colors3,
    "basis" : "xy",
}

reactor_universe.plot(**plot_args3, legend=True)
plt.xlabel("x [cm]", fontsize=16)
plt.ylabel("y [cm]", fontsize=16)
plt.title("Cross-section of Reactor Core (xy-plane)", fontsize=18, weight="bold")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.gca().set_aspect('equal', adjustable='box')  # Keep aspect ratio square
plt.tight_layout()
plt.savefig("ccore_xy_plot.png", dpi=300)
plt.show()
"""
geometry = openmc.Geometry(reactor_universe)
geometry.export_to_xml("multi-group_cross-sections/maia_geometry.xml")





