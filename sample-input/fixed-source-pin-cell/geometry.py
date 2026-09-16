import numpy as np
import openmoc


# ============================================================
# SPATIAL GRID
# ============================================================

xmin_value = -2.0
xmax_value =  2.0
ymin_value = -2.0
ymax_value =  2.0

# Number of spatial cells
NX = 80
NY = 80

dx = (xmax_value - xmin_value) / NX
dy = (ymax_value - ymin_value) / NY

# Cell-centre coordinates
x_centers = np.linspace(
    xmin_value + dx / 2.0,
    xmax_value - dx / 2.0,
    NX
)

y_centers = np.linspace(
    ymin_value + dy / 2.0,
    ymax_value - dy / 2.0,
    NY
)


# ============================================================
# ONE-GROUP NON-FISSILE MATERIAL
# ============================================================

material = openmoc.Material(name='non-fissile material')

material.setNumEnergyGroups(1)

# Same one-group material physics used in William's first case
material.setSigmaT(
    np.array([1.0])
)

material.setSigmaS(
    np.array([0.5])
)

material.setSigmaF(
    np.array([0.0])
)

material.setNuSigmaF(
    np.array([0.0])
)

material.setChi(
    np.array([1.0])
)


# ============================================================
# GRID SURFACES
# ============================================================

x_edges = np.linspace(
    xmin_value,
    xmax_value,
    NX + 1
)

y_edges = np.linspace(
    ymin_value,
    ymax_value,
    NY + 1
)

xplanes = []
yplanes = []

# Create x surfaces
for i, x in enumerate(x_edges):

    surface = openmoc.XPlane(
        x=x,
        name='xplane-{}'.format(i)
    )

    # Outer boundary is vacuum
    if i == 0 or i == NX:
        surface.setBoundaryType(openmoc.VACUUM)

    xplanes.append(surface)


# Create y surfaces
for j, y in enumerate(y_edges):

    surface = openmoc.YPlane(
        y=y,
        name='yplane-{}'.format(j)
    )

    # Outer boundary is vacuum
    if j == 0 or j == NY:
        surface.setBoundaryType(openmoc.VACUUM)

    yplanes.append(surface)


# ============================================================
# CREATE ONE CELL PER GRID ELEMENT
# ============================================================

root_universe = openmoc.Universe(name='root universe')

cells = []

for j in range(NY):

    row = []

    for i in range(NX):

        cell = openmoc.Cell(
            name='cell-{}-{}'.format(i, j)
        )

        cell.setFill(material)

        # x_i < x < x_(i+1)
        cell.addSurface(
            +1,
            xplanes[i]
        )

        cell.addSurface(
            -1,
            xplanes[i + 1]
        )

        # y_j < y < y_(j+1)
        cell.addSurface(
            +1,
            yplanes[j]
        )

        cell.addSurface(
            -1,
            yplanes[j + 1]
        )

        root_universe.addCell(cell)

        row.append(cell)

    cells.append(row)


# ============================================================
# GEOMETRY
# ============================================================

geometry = openmoc.Geometry()

geometry.setRootUniverse(
    root_universe
)

geometry.initializeFlatSourceRegions()