#!/usr/bin/env python

from __future__ import absolute_import

import numpy

import dynamic_wave

def bc_inflow(state, dim, t, qbc, auxbc, num_ghost):
    qbc[0, :num_ghost] = 10.0
    qbc[1, :num_ghost] = qbc[1, num_ghost]

def bc_outflow(state, dim, t, qbc, auxbc, num_ghost):
    qbc[0, -num_ghost:] = 7.0
    qbc[1, -num_ghost:] = qbc[1, -num_ghost - 1]

def setup(kernel_language='Python', solver_type='classic', use_petsc=False,
          outdir='./_output'):

    if use_petsc:
        import clawpack.petclaw as pyclaw
    else:
        from clawpack import pyclaw

    if kernel_language == 'Fortran':
        raise NotImplementedError("A fortran version of the Riemann solver is not available yet.")
    elif kernel_language == 'Python':
        solver = pyclaw.ClawSolver1D(dynamic_wave.dynamic_wave_1d)
        solver.kernel_language = 'Python'
    solver.order = 2
    solver.limiters = pyclaw.limiters.tvd.vanleer
    solver.fwave = True
    solver.num_waves = 2
    solver.num_eqn = 2
    solver.bc_lower[0] = pyclaw.BC.custom
    solver.user_bc_lower = bc_inflow
    solver.bc_upper[0] = pyclaw.BC.custom
    solver.user_bc_upper = bc_outflow
    solver.aux_bc_lower[0] = pyclaw.BC.extrap
    solver.aux_bc_upper[0] = pyclaw.BC.extrap

    xlower = 0.0
    xupper = 1e3
    x = pyclaw.Dimension(xlower, xupper, 500, name='x')
    domain = pyclaw.Domain(x)
    state = pyclaw.State(domain, 2, 1)

    # Gravitational constant
    state.problem_data['grav'] = 9.81
    state.problem_data['dry_tolerance'] = 1e-3
    state.problem_data['width'] = 1.0
    state.problem_data['mannings'] = 0.0

    xc = state.grid.x.centers
    state.aux[0, :] = ((x.centers >= 125.0) * (x.centers <= 875.0)) * 5.0 * (numpy.sin(numpy.pi * ((x.centers - 125.0) / 750.0)))**2
    state.q[0, :] = 10.0 * numpy.ones(x.num_cells)
    state.q[1, :] = 20.0 * numpy.ones(x.num_cells)

    claw = pyclaw.Controller()
    claw.keep_copy = True
    claw.tfinal = 500.0
    claw.num_output_times = 50
    claw.solution = pyclaw.Solution(state, domain)
    claw.solver = solver
    claw.setplot = setplot
    claw.write_aux_init = True

    if outdir is not None:
        claw.outdir = outdir
    else:
        claw.output_format = None

    return claw


#--------------------------
def setplot(plotdata):
#--------------------------
    """ 
    Specify what is to be plotted at each frame.
    Inumpyut:  plotdata, an instance of visclaw.data.ClawPlotData.
    Output: a modified version of plotdata.
    """ 
    plotdata.clearfigures()  # clear any old figures,axes,items data

    # Plot variables
    def slope(current_data):
        return current_data.aux[0, :]

    def h(current_data, width=1.0):
        return current_data.q[0, :] / width

    def eta(current_data):
        return h(current_data) + slope(current_data)

    def velocity(current_data):
        return current_data.q[1, :] / current_data.q[0, :]

    rgb_converter = lambda triple: [float(rgb) / 255.0 for rgb in triple]

    # Figure for depth
    plotfigure = plotdata.new_plotfigure(name='Depth', figno=0)

    # Axes for water depth
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = [0.0, 1e3]
    plotaxes.ylimits = [-1.1, 11.0]
    plotaxes.title = 'Water Depth'
    plotaxes.axescmd = 'subplot(211)'

    plotitem = plotaxes.new_plotitem(plot_type='1d_fill_between')
    plotitem.plot_var = eta
    plotitem.plot_var2 = slope
    plotitem.color = rgb_converter((67,183,219))

    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = slope
    plotitem.color = 'k'

    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = eta
    plotitem.color = 'k'

    # Axes for velocity
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.axescmd = 'subplot(212)'
    plotaxes.xlimits = [0.0, 1e3]
    plotaxes.ylimits = [-15.0, 15.0]
    plotaxes.title = 'Velocity'

    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = velocity
    plotitem.color = 'b'
    plotitem.kwargs = {'linewidth':3}
    
    return plotdata


if __name__=="__main__":
    from clawpack.pyclaw.util import run_app_from_main
    output = run_app_from_main(setup,setplot)
