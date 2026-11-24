###############################################################################
#                          Start with
###############################################################################
# conda activate openmoc-env
# last run: python core_run.py -a 4 -p 2 -s 5 -l 5 -t 4 -i 1000 -c 1e-5

# Example runs:
#
# 1) Run with OpenMOC defaults
#    python core_run.py
# 2) Small/cheap 3D test
#    python core_run.py -a 4 -p 2 -s 0.2 -l 1.0 -t 2 -i 50
# 3) Moderate run
#    python core_run.py -a 8 -p 4 -s 0.1 -l 0.5 -t 4 -i 100 -c 1e-5
# 4) Finer angular resolution
#    python core_run.py -a 16 -p 6 -s 0.1 -l 0.5 -t 4 -i 200 -c 1e-5
# 5) Finer spatial/axial sampling
#    python core_run.py -a 16 -p 6 -s 0.05 -l 0.25 -t 8 -i 300 -c 1e-5

import openmoc
import openmoc.log as log
from openmoc import plotter as plotter
from openmoc.options import Options
from geometry import geometry
from pathlib import Path
import numpy as np

###############################################################################
#                          Main Simulation Parameters (default)
###############################################################################
# The most important runtime parameters:
#-t / --num-omp-threads → number of CPU threads used = 1
#-a / --num-azim → number of azimuthal angles = 4
#-s / --azim-spacing → track spacing in cm = 0.1
#-i / --max-iters → maximum number of transport/source iterations = 1000
#-c / --tolerance → convergence tolerance = 1E-5
#-p / --num-polar → number of polar angles, mainly for 3D problems = 6
#-l / --z-spacing → axial ray spacing, also for 3D problems. = 1.5
#-b / --num-threadblocks → Related with GPU : 64 blocks
#-g / --num-threads-per-block → Related with GPU : 64 CUDA threads per block

options = Options()
openmoc.set_line_length(120)

num_threads = options.num_omp_threads
azim_spacing = options.azim_spacing
num_azim = options.num_azim
polar_spacing = options.polar_spacing
num_polar = options.num_polar
tolerance = options.tolerance
max_iters = options.max_iters

###############################################################################
#                          Results file
###############################################################################

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "log"
LOG_DIR.mkdir(parents=True, exist_ok=True)

###############################################################################
#                          Creating the TrackGenerator
###############################################################################

log.py_printf('NORMAL', 'Initializing the track generator...')

quad = openmoc.GLPolarQuad() # Use Gauss–Legendre quadrature to choose the polar angles and weights.
quad.setNumPolarAngles(num_polar)
quad.setNumAzimAngles(num_azim)

track_generator = openmoc.TrackGenerator3D(geometry, num_azim, num_polar,
                                          azim_spacing, polar_spacing)
track_generator.setQuadrature(quad) # Tell the track generator to use the quadrature we have defined
track_generator.setNumThreads(num_threads)
track_generator.setSegmentFormation(openmoc.OTF_STACKS) # Construct 3D segments on the fly to control memory usage
track_generator.setSegmentationZones([-120.0, 120.0]) 
track_generator.generateTracks()

###############################################################################
#                            Running a Simulation
###############################################################################

solver = openmoc.CPUSolver(track_generator)
solver.setNumThreads(num_threads)
solver.setConvergenceThreshold(tolerance)
solver.computeEigenvalue(max_iters)
solver.printTimerReport()

###############################################################################
#                  Export FSR fission-rate information
###############################################################################

num_fsrs = geometry.getNumFSRs()

fission_rates = np.asarray(
    solver.computeFSRFissionRates(num_fsrs),
    dtype=float
)

fsr_id = np.arange(num_fsrs)

x = np.empty(num_fsrs)
y = np.empty(num_fsrs)
z = np.empty(num_fsrs)
volume = np.empty(num_fsrs)

for i in range(num_fsrs):
    centroid = geometry.getFSRCentroid(i)

    x[i] = centroid.getX()
    y[i] = centroid.getY()
    z[i] = centroid.getZ()
    volume[i] = solver.getFSRVolume(i)

data = np.column_stack((fsr_id, x, y, z, volume, fission_rates))

tag = (
    f"a{num_azim}_p{num_polar}_"
    f"s{azim_spacing:g}_l{polar_spacing:g}"
).replace(".", "p")

output_file = LOG_DIR / f"fsr_fission_{tag}.csv"

np.savetxt(
    output_file,
    data,
    delimiter=",",
    header=(
        "fsr_id,x_cm,y_cm,z_cm,"
        "volume_cm3,fission_rate"
    ),
    comments=""
)

log.py_printf(
    'NORMAL',
    'FSR fission-rate data written to %s',
    str(output_file)
)

###############################################################################
#                             Generating Plots
###############################################################################

openmoc.log.py_printf('NORMAL', 'Plotting data...')
#plotter.plot_quadrature(solver)
plotter.plot_tracks(track_generator, get_figure=True, plot_3D=True)
plotter.plot_tracks(track_generator, get_figure=True)
plotter.plot_segments(track_generator, get_figure=True, plot_3D=True)
plotter.plot_segments(track_generator, get_figure=True)
#plotter.plot_materials(geometry, gridsize=500, plane='xy', offset=0.)
#plotter.plot_cells(geometry, gridsize=500, plane='xy', offset=0.)
plotter.plot_flat_source_regions(geometry, gridsize=1000, plane='xy', centroids=True, get_figure=True)
plotter.plot_fission_rates(solver, norm=True, gridsize=1000, plane='xy', get_figure=True)
#plotter.plot_spatial_fluxes(solver, energy_groups=[1,2,3,4,5,6,7], plane='xy', offset=0.)
#plotter.plot_energy_fluxes(solver, fsrs=range(geometry.getNumFSRs()))

openmoc.log.py_printf('TITLE', 'Finished')