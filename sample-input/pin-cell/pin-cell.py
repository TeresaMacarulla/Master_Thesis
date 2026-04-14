import openmoc
from geometry import *
from openmoc import plotter as plotter

###############################################################################
#                          Main Simulation Parameters
###############################################################################

options = openmoc.options.Options()

num_threads = options.num_omp_threads
azim_spacing = options.azim_spacing
num_azim = options.num_azim
num_polar = options.num_polar
tolerance = options.tolerance
max_iters = options.max_iters

# The most important runtime parameters:
#-t / --num-omp-threads → number of CPU threads used
#-a / --num-azim → number of azimuthal angles
#-s / --azim-spacing → track spacing in cm
#-i / --max-iters → maximum number of transport/source iterations
#-c / --tolerance → convergence tolerance
#-p / --num-polar → number of polar angles, mainly for 3D problems
#-l / --z-spacing → axial ray spacing, also for 3D problems.

###############################################################################
#                          Creating the TrackGenerator
###############################################################################

openmoc.log.py_printf('NORMAL', 'Initializing the track generator...')

track_generator = openmoc.TrackGenerator(geometry, num_azim, azim_spacing)
track_generator.setNumThreads(num_threads)
track_generator.setZCoord(0.1)
track_generator.generateTracks()


###############################################################################
#                            Running a Simulation
###############################################################################

solver = openmoc.CPULSSolver(track_generator)
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
plotter.plot_segments(track_generator)
plotter.plot_materials(geometry, gridsize=500, plane='xy', offset=0.)
plotter.plot_cells(geometry, gridsize=500, plane='xy', offset=0.)
plotter.plot_flat_source_regions(geometry, gridsize=500, plane='xy', offset=0.)
plotter.plot_spatial_fluxes(solver, energy_groups=[1,2,3,4,5,6,7], \
  plane='xy', offset=0.)
plotter.plot_energy_fluxes(solver, fsrs=range(geometry.getNumFSRs()))

openmoc.log.py_printf('TITLE', 'Finished')
