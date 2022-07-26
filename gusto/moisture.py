from firedrake import exp, dx, conditional


def condensation(test, Q, D, parameters):
    alpha = parameters.alpha
    q_0 = parameters.q_0
    H = parameters.H
    tau = parameters.tau
    q_s = q_0 * exp(-alpha*(D-H)/H)
    expr = conditional(Q > q_s, (Q - q_s) * (Q - q_s)/tau, 0)
    return test * expr * dx


def evaporation(test, Q, parameters):
    q_g = parameters.q_g
    tau_e = parameters.tau_e
    expr = conditional(q_g > Q, (q_g - Q) * (q_g - Q)/tau_e, 0)
    return test * expr * dx
