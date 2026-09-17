import os
import numpy as np
import matplotlib.pyplot as plt

import openmoc
import openmoc.plotter as plotter

from geometry import (
    geometry,
    NX,
    NY,
    dx,
    dy,
    xmin_value,
    xmax_value,
    ymin_value,
    ymax_value,
    x_centers,
    y_centers
)

#conda activate openmoc-env
###############################################################################
#              WORKFLOW: 2D GAUSSIAN RANDOM FIELD FIXED SOURCE
###############################################################################
#
# The objective of this script is to generate one training-data pair:
#
#                       Q(x,y)  --->  phi(x,y)
#
# where Q(x,y) is a spatially varying fixed neutron source generated from a
# Gaussian Random Field (GRF), and phi(x,y) is the corresponding scalar
# neutron flux calculated by OpenMOC.
#
#
# 1. SPATIAL DISCRETIZATION
#
# The 4 cm x 4 cm domain is divided into an NX x NY regular grid.
#
# Each grid cell corresponds to one OpenMOC Flat Source Region (FSR).
# Therefore, each FSR will receive one constant source value Q_ij.
#
#
# 2. GRID COORDINATES
#
# The center of every grid cell is stored as a 2D coordinate:
#
#       r_k = (x_k, y_k)
#
# Although the source is two-dimensional, the NX x NY grid is temporarily
# flattened into a list of N = NX*NY spatial positions. 
#
#
# 3. GAUSSIAN RANDOM FIELD COVARIANCE
#
# The spatial correlation between every pair of points r_a and r_b is
# described using the squared-exponential covariance kernel:
#                      
#       C_ab = sigma^2 exp[(-|r_a - r_b|^2)/(2*l^2)]                         
#
# where:
#
#       sigma^2 = variance of the source values
#       l       = spatial correlation length scale
#
# Since there are N spatial points, the covariance matrix has dimensions:
#
#       C.shape = (N, N)
#
#
# 4. NUMERICAL STABILIZATION
#
# A very small number is added to the diagonal of the covariance matrix:
#
#       C -> C + epsilon*I
#
# This "jitter" compensates for floating-point rounding errors and makes the
# Cholesky factorization numerically more stable. The value epsilon = 1e-10
# is sufficiently small that it does not meaningfully change the GRF.
#
#
# 5. GENERATING ONE GRF REALIZATION
#
# The covariance matrix is factorized using the Cholesky decomposition:
#
#       C = L L^T
#
# A vector Z of N independent standard-normal random values is generated:
#
#       Z ~ N(0, I)
#
# The spatially correlated source is then obtained as:
#
#       Q_flat = M + L Z
#
# where M is the mean-source vector.
#
# For a N=NX*NY grid:
#
#       L.shape      = (N, N)
#       Z.shape      = (N,)
#       Q_flat.shape = (N,)
#
#
# 6. RESTORING THE 2D SOURCE FIELD
#
# Q_flat contains one source value for each spatial location but is stored as
# a one-dimensional vector. It is reshaped back to the original grid:
#
#       Q = Q_flat.reshape(NY, NX)
#
# giving:
#
#       Q.shape = (20, 20)
#
# Thus Q[j,i] represents the source value in the grid cell centered at:
#
#       (x_i, y_j)
#
#
# 7. MAPPING THE GRF TO OPENMOC
#
# Each rectangular grid cell is represented by one FSR. For each FSR we:
#
#   a) obtain a representative position (x,y),
#   b) identify the corresponding grid indices (i,j),
#   c) obtain Q[j,i],
#   d) assign this value using:
#
#       solver.setFixedSourceByFSR(fsr_id, 1, Q[j,i])
#
# Therefore, OpenMOC sees a piecewise-constant approximation to the continuous
# Gaussian random field.
#
#
# 8. NEUTRON TRANSPORT SOLUTION
#
# OpenMOC solves the fixed-source transport problem including scattering:
#
#       Q(x,y) ---> phi(x,y)
#
# using:
#
#       solver.computeSource(...)
#
# The resulting scalar flux is extracted for every FSR.
#
#
# 9. RECONSTRUCTING THE FLUX FIELD
#
# The scalar flux values are mapped back onto the same NX x NY grid, producing:
#
#       Phi.shape = (NY, NX)
#
# Therefore one complete machine-learning training sample consists of:
#
#       input:   Q   shape = (NY, NX)
#       output:  Phi shape = (NY, NX)
#
# i.e.
#
#                    Q(x,y) ---> phi(x,y)
#
###############################################################################

# ============================================================
# GRF PARAMETERS
# ============================================================

mean = 50.0
variance = 2.0
length_scale = 1.0
random_seed = 1234

# ============================================================
# BUILD THE 2D GRF
# ============================================================

# Create all grid-centre coordinates
X, Y = np.meshgrid(
    x_centers,
    y_centers
)

# Shape:
# (NY, NX, 2)
coords = np.column_stack(
    (
        X.ravel(),
        Y.ravel()
    )
)

num_points = coords.shape[0]

# ------------------------------------------------------------
# Mean vector
# ------------------------------------------------------------

M = np.full(
    num_points,
    mean
)

# ------------------------------------------------------------
# Squared-exponential covariance matrix
# ------------------------------------------------------------

# coords[:, None, :] has shape (N,1,2)
# coords[None, :, :] has shape (1,N,2)
#
# Difference gives all pairwise coordinate differences.

diff = (
    coords[:, None, :]
    -
    coords[None, :, :]
)

distance_squared = np.sum(
    diff**2,
    axis=2
)

C = (
    variance
    *
    np.exp(
        -distance_squared
        /
        (2.0 * length_scale**2)
    )
)


# Small numerical jitter for stable Cholesky decomposition
epsilon = 1e-10

C = (
    C
    +
    epsilon * np.eye(num_points)
)


# ------------------------------------------------------------
# Draw one GRF realization
# ------------------------------------------------------------

rng = np.random.default_rng(
    random_seed
)

L_chol = np.linalg.cholesky(
    C
)

Z = rng.standard_normal(
    num_points
)

Q_flat = M + L_chol @ Z


# Negative neutron-source values are nonphysical.
Q_flat = np.clip(
    Q_flat,
    0.0,
    None
)


# Reshape to a 2D field
Q = Q_flat.reshape(
    NY,
    NX
)


# ============================================================
# TRACK GENERATION
# ============================================================

num_azim = 32
azim_spacing = 0.05

track_generator = openmoc.TrackGenerator(
    geometry,
    num_azim,
    azim_spacing
)

track_generator.setNumThreads(1)

track_generator.generateTracks()


# ============================================================
# SOLVER
# ============================================================

solver = openmoc.CPUSolver(
    track_generator
)

solver.setNumThreads(1)

solver.setConvergenceThreshold(
    1.0e-5
)


# ============================================================
# ASSIGN THE GRF TO THE FSRs
# ============================================================

num_fsrs = geometry.getNumFSRs()

print( "Number of FSRs:", num_fsrs )
print( "Expected number:", NX * NY )

# Store the grid indices associated with each FSR.
# These will later be reused to reconstruct the flux matrix.
fsr_i = np.zeros(num_fsrs, dtype=int)
fsr_j = np.zeros(num_fsrs, dtype=int)


for fsr_id in range(num_fsrs):

    # OpenMOC provides a representative point inside each FSR
    point = geometry.getFSRPoint(fsr_id)

    x = point.getX()
    y = point.getY()

    # Identify the Cartesian grid cell containing this FSR
    i = int((x - xmin_value) / dx)
    j = int((y - ymin_value) / dy)


    # Protect against floating-point boundary issues
    i = np.clip(i, 0, NX - 1)
    j = np.clip(j, 0, NY - 1)

    # Save the mapping FSR -> grid cell
    fsr_i[fsr_id] = i
    fsr_j[fsr_id] = j

    # Assign the source to this FSR

    solver.setFixedSourceByFSR(
        fsr_id,
        1,
        Q[j,i]
    )


# ============================================================
# SOLVE THE FIXED-SOURCE TRANSPORT PROBLEM
# ============================================================

solver.computeSource(1000)
solver.printTimerReport()


# ============================================================
# EXTRACT AND RECONSTRUCT SCALAR FLUX
# ============================================================

Phi = np.zeros((NY, NX))

FSR_ID = np.zeros(
    (NY, NX),
    dtype=int
)


for fsr_id in range(num_fsrs):

    # Retrieve the grid indices already calculated
    # before solving the transport problem
    i = fsr_i[fsr_id]
    j = fsr_j[fsr_id]

    # Extract scalar flux from OpenMOC
    flux = solver.getFlux(
        fsr_id,
        1
    )

    # Map the FSR result back onto the Cartesian grid
    Phi[j, i] = flux

    # Save the FSR identifier corresponding to this grid cell
    FSR_ID[j, i] = fsr_id


# ============================================================
# SAVE TRAINING PAIR
# ============================================================
# Grid-size label used in output filenames
grid_tag = f"{NX}x{NY}"

os.makedirs(
    'log',
    exist_ok=True
)

np.savez_compressed(
    f'log/Q_Phi-mean{mean}-variance{variance}-l{length_scale}-grid{grid_tag}.npz',
    Q=Q,
    phi=Phi,
    x=x_centers,
    y=y_centers,
    fsr_ids=FSR_ID,
    mean=mean,
    variance=variance,
    length_scale=length_scale,
    random_seed=random_seed
)

# ============================================================
# PLOT Q(x,y)
# ============================================================

plt.figure()
plt.imshow(
    Q,
    origin='lower',
    extent=[
        xmin_value,
        xmax_value,
        ymin_value,
        ymax_value
    ],
    aspect='equal'
)
plt.colorbar(
    label='Q(x,y)'
)
plt.xlabel(
    'x [cm]'
)
plt.ylabel(
    'y [cm]'
)
plt.title(
    f'2D Gaussian Random Field Source ({NX}x{NY} grid)'
)
plt.tight_layout()
plt.savefig(
    f'plots/Q-GRF-mean{mean}-variance{variance}-l{length_scale}-grid{grid_tag}.png',
    dpi=200
)

# ============================================================
# PLOT PHI(x,y)
# ============================================================

plt.figure()
plt.imshow(
    Phi,
    origin='lower',
    extent=[
        xmin_value,
        xmax_value,
        ymin_value,
        ymax_value
    ],
    aspect='equal'
)
plt.colorbar(
    label='Scalar flux'
)
plt.xlabel(
    'x [cm]'
)
plt.ylabel(
    'y [cm]'
)
plt.title(
    f'OpenMOC Scalar Flux ({NX}x{NY} grid)'
)
plt.tight_layout()
plt.savefig(
    f'plots/phi-GRF-mean{mean}-variance{variance}-l{length_scale}-grid{grid_tag}.png',
    dpi=200
)

# ============================================================
# OPTIONAL OPENMOC PLOTS
# ============================================================

plotter.plot_flat_source_regions(
    geometry,
    gridsize=300
)

plotter.plot_spatial_fluxes(
    solver,
    energy_groups=[1]
)