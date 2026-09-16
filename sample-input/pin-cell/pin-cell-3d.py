###############################################################################
#                          Start with
###############################################################################
# conda activate openmoc-env

# Example runs:
#
# 1) Run with OpenMOC defaults
#    python pin-cell-3d.py
# 2) Small/cheap 3D test
#    python pin-cell-3d.py -a 4 -p 2 -s 0.2 -l 1.0 -t 2 -i 50
# 3) Moderate run
#    python pin-cell-3d.py -a 8 -p 4 -s 0.1 -l 0.5 -t 4 -i 100 -c 1e-5
# 4) Finer angular resolution
#    python pin-cell-3d.py -a 16 -p 6 -s 0.1 -l 0.5 -t 4 -i 200 -c 1e-5
# 5) Finer spatial/axial sampling
#    python pin-cell-3d.py -a 16 -p 6 -s 0.05 -l 0.25 -t 8 -i 300 -c 1e-5

import openmoc
from geometry import *
from openmoc import plotter as plotter

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

options = openmoc.options.Options()

num_threads = options.num_omp_threads
azim_spacing = options.azim_spacing
num_azim = options.num_azim
polar_spacing = options.polar_spacing
num_polar = options.num_polar
tolerance = options.tolerance
max_iters = options.max_iters


###############################################################################
#                          Creating the TrackGenerator
###############################################################################

openmoc.log.py_printf('NORMAL', 'Initializing the track generator...')

track_generator = openmoc.TrackGenerator3D(geometry, num_azim, num_polar,
                                          azim_spacing, polar_spacing)
track_generator.setNumThreads(num_threads)
track_generator.setSegmentFormation(openmoc.OTF_STACKS)
track_generator.setSegmentationZones([-2.0, 2.0])
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
#                             Generating Plots
###############################################################################

openmoc.log.py_printf('NORMAL', 'Plotting data...')
plotter.plot_quadrature(solver)
plotter.plot_tracks(track_generator)
plotter.plot_tracks(track_generator, plot_3D=True)
plotter.plot_materials(geometry, gridsize=500, plane='xy', offset=0.)
plotter.plot_cells(geometry, gridsize=500, plane='xy', offset=0.)
plotter.plot_flat_source_regions(geometry, gridsize=500, plane='xy', offset=0.)
plotter.plot_spatial_fluxes(solver, energy_groups=[1,2,3,4,5,6,7], \
  plane='xy', offset=0.)
plotter.plot_energy_fluxes(solver, fsrs=range(geometry.getNumFSRs()))

openmoc.log.py_printf('TITLE', 'Finished')
