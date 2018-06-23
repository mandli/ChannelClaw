#!/usr/bin/env python

import numpy

def dynamic_wave_1d(q_l, q_r, aux_l, aux_r, problem_data):
    r"""Riemann solver for dynamic wave equation (St. Venanant)

    *problem_data* should contain:
     - *g* - (float) Gravitational constant
     - *dry_tolerance* - (float) Set velocities to zero if h is below this
       tolerance.
     - *I1* - (func) Function that calculate the pressure

    -> the aux array (index == 0) contains the slope of the channel S_0
    """

    g = problem_data['grav']
    dry_tolerance = problem_data['dry_tolerance']
    width = problem_data['width']

    I1 = lambda h: h**2 * width / 2.0
    I2 = lambda h: numpy.zeros(h.shape)
    beta = lambda h: width
    h = lambda A: A / width

    num_rp = q_l.shape[1]
    num_eqn = 2
    num_waves = 2

    # Output arrays
    fwave = numpy.empty( (num_eqn, num_waves, num_rp) )
    s = numpy.empty( (num_waves, num_rp) )
    amdq = numpy.zeros( (num_eqn, num_rp) )
    apdq = numpy.zeros( (num_eqn, num_rp) )

    # Extract state
    h_l = h(q_l[0, :])
    h_r = h(q_r[0, :])
    h_bar = 0.5 * (h_l + h_r)
    u_l = numpy.where(q_l[0, :] > dry_tolerance,
                   q_l[1, :] / q_l[0, :], 0.0)
    u_r = numpy.where(q_r[0, :] > dry_tolerance,
                   q_r[1, :] / q_r[0, :], 0.0)
    phi_l = h_l * u_l**2 + g * I1(h_l)
    phi_r = h_r * u_r**2 + g * I1(h_r)
    A_bar = 0.5 * (q_l[0, :] + q_r[0, :])

    # Compute Speeds
    c_l = numpy.sqrt(g * q_l[0, :] / beta(h_l))
    c_r = numpy.sqrt(g * q_r[0, :] / beta(h_r))
    u_hat = (c_l * u_l + c_r * u_r) / (c_l + c_r)
    c_hat = numpy.sqrt(g * A_bar / beta(h_bar))
    s[0, :] = numpy.amin(numpy.vstack((u_l - c_l, u_hat - c_hat)), axis=0)
    s[1, :] = numpy.amax(numpy.vstack((u_r + c_r, u_hat + c_hat)), axis=0)

    # Friction effects are in the source term step as well as the I2 term
    # Note that depending on the definition of S the value here may be the 
    # opposite of what is usually defined.
    delta_1 = q_r[1, :] - q_l[1, :]
    delta_2 = phi_r - phi_l + g * A_bar * (aux_r[0, :] - aux_l[0, :])

    beta_1 = (s[1, :] * delta_1 - delta_2) / (s[1, :] - s[0, :])
    beta_2 = (delta_2 - s[0, :] * delta_1) / (s[1, :] - s[0, :])

    fwave[0, 0, :] = beta_1
    fwave[1, 0, :] = beta_1 * s[0, :]
    fwave[0, 1, :] = beta_2
    fwave[1, 1, :] = beta_2 * s[1, :]

    for m in range(num_eqn):
        for mw in range(num_waves):
            amdq[m, :] += (s[mw, :] < 0.0) * fwave[m, mw, :]
            apdq[m, :] += (s[mw, :] > 0.0) * fwave[m, mw, :]

            amdq[m, :] += (s[mw, :] == 0.0) * fwave[m, mw, :] * 0.5
            apdq[m, :] += (s[mw, :] == 0.0) * fwave[m, mw, :] * 0.5

    return fwave, s, amdq, apdq