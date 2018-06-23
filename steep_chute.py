#!/usr/bin/env python

from __future__ import absolute_import

import numpy

import dynamic_wave

def source_step(solver, state, dt):
    """Handle the RHS terms

    Q_t = -g A S_f + g I_2
    """
    g = state.problem_data['grav']
    width = state.problem_data['width']
    h = lambda A: A / width

    # Hydraulic radius = A / P where P is the wetted perimiter
    R = state.q[0, :] / (2.0 * h(state.q[0, :]) + width)
    K = state.q[0, :] * R**(2.0 / 3.0) / state.problem_data['mannings']
    state.q[1, :] -= dt * g * state.q[0, :] * state.q[1, :] * numpy.linalg.norm(state.q[1, :], ord=2) / K**2

    # I2 term is zero in this case

def bc_inflow(state, dim, t, qbc, auxbc, num_ghost):
    qbc[0, :num_ghost] = qbc[0, num_ghost]
    qbc[1, :num_ghost] = 20.0

def bc_outflow(state, dim, t, qbc, auxbc, num_ghost):
    qbc[0, -num_ghost:] = 2.5 * 8.0 # Width of channel is 8.0 meters
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
    solver.step_source = source_step
    solver.source_split = 1

    xlower = 0.0
    xupper = 1e3
    x = pyclaw.Dimension(xlower, xupper, 500, name='x')
    domain = pyclaw.Domain(x)
    state = pyclaw.State(domain, 2, 1)

    # Gravitational constant
    state.problem_data['grav'] = 9.81
    state.problem_data['dry_tolerance'] = 1e-3
    state.problem_data['width'] = 8.0
    state.problem_data['mannings'] = 0.015

    xc = state.grid.x.centers
    state.aux[0, :] =   (xc < 300.0) * (-0.002 * (xc - 300.0) + 3.1) \
                      + (xc >= 300.0) * (xc < 600.0) * (-0.009 * (xc - 600.0) + 0.4) \
                      + (xc >= 600.0) * (-0.001 * (xc - 1000.0))
    state.q[0, :] = 4.5 * 8.0 * numpy.ones(x.num_cells)
    state.q[1, :] = 20.0 * numpy.ones(x.num_cells)

    claw = pyclaw.Controller()
    claw.keep_copy = True
    claw.tfinal = 1000.0
    claw.num_output_times = 100
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

    def h(current_data, width=8.0):
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
    plotaxes.ylimits = [0.0, 6.0]
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
    plotaxes.ylimits = [-1.0, 80.0]
    plotaxes.title = 'Discharge'

    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    # plotitem.plot_var = velocity
    plotitem.plot_var = 1
    plotitem.color = 'b'
    plotitem.kwargs = {'linewidth':3}
    
    return plotdata


if __name__=="__main__":
    from clawpack.pyclaw.util import run_app_from_main
    output = run_app_from_main(setup,setplot)
