import sympy as sp
import pdb
import spb
import matplotlib.pyplot as plt
import numpy as np

Omega = 1.74e-4
R = 71.4e6
Lx = 7e7
nx = 256

r = sp.symbols('r')
rstar = Lx/2
delta = 2*Lx/nx

###################################

# delta = 2*Lx/nx

# fexpr = 2*Omega*(1-0.5*r**2/R**2)
# left_val = fexpr.subs(r, rstar-delta)
# right_val = 2*Omega
# left_diff_val = sp.diff(fexpr, r).subs(r, rstar-delta)
# left_diff2_val = sp.diff(fexpr, r, 2).subs(r, rstar-delta)

# degree = 5
# a = sp.symbols('a_0:6')

# P = a[0] + a[1]*r + a[2]*r**2 + a[3]*r**3 + a[4]*r**4 + a[5]*r**5

# eqns = [
#     P.subs(r, rstar-delta) - left_val,
#     P.subs(r, rstar+delta) - right_val,
#     sp.diff(P, r).subs(r, rstar-delta) - left_diff_val,
#     sp.diff(P, r).subs(r, rstar+delta),
#     sp.diff(P, r, 2).subs(r, rstar-delta) - left_diff2_val,
#     sp.diff(P, r, 2).subs(r, rstar+delta)
# ]

# sol = sp.solve(eqns, a)
# P_smooth = P.subs(sol)

# f_smooth = sp.Piecewise(
#     (fexpr, r<rstar-delta),
#     (P_smooth, (rstar-delta<=r) & (r<=rstar+delta)),
#     (right_val, rstar+delta<r)
# )

# plot = sp.plot(f_smooth, (r, rstar-5*delta, rstar+5*delta))
# plot.save(f'/data/home/sh1293/firedrake-real-opt_jun25/src/gusto/examples/shallow_water/smooth_pv_5.pdf')
# print(f'Plot made:\n /data/home/sh1293/firedrake-real-opt_jun25/src/gusto/examples/shallow_water/smooth_pv_5.pdf')


###################################
# function from full script

def smooth_f_profile(degree, delta, Omega=Omega, R=R, Lx=Lx, nx=nx):
    delta *= Lx/nx
    r = sp.symbols('r')
    fexpr = 2*Omega*(1-0.5*r**2/R**2)
    rstar = Lx/2
    left_val = fexpr.subs(r, rstar-delta)
    right_val = 2*Omega
    left_diff_val = sp.diff(fexpr, r).subs(r, rstar-delta)
    left_diff2_val = sp.diff(fexpr, r, 2).subs(r, rstar-delta)

    a = sp.symbols(f'a_0:{degree+1}')
    P = a[0]
    for i in range(1, degree+1):
        P += a[i]*r**i

    if degree == 3:
        eqns = [
            P.subs(r, rstar-delta) - left_val,
            P.subs(r, rstar+delta) - right_val,
            sp.diff(P, r).subs(r, rstar-delta) - left_diff_val,
            sp.diff(P, r).subs(r, rstar+delta)
        ]
    elif degree == 5:
        eqns = [
            P.subs(r, rstar-delta) - left_val,
            P.subs(r, rstar+delta) - right_val,
            sp.diff(P, r).subs(r, rstar-delta) - left_diff_val,
            sp.diff(P, r).subs(r, rstar+delta),
            sp.diff(P, r, 2).subs(r, rstar-delta) - left_diff2_val,
            sp.diff(P, r, 2).subs(r, rstar+delta)
        ]
    else:
        print('do not have BCs for this degree')

    sol = sp.solve(eqns, a)

    P_smooth = P.subs(sol)

    f_smooth = sp.Piecewise(
        (fexpr, r<rstar-delta),
        (P_smooth, (rstar-delta<=r) & (r<=rstar+delta)),
        (right_val, rstar+delta<r)
    )

    return f_smooth


smooth3 = smooth_f_profile(degree=3, delta=2)
smooth5 = smooth_f_profile(degree=5, delta=2)

s3f = sp.lambdify(r, smooth3)
s5f = sp.lambdify(r, smooth5)

rarray = np.linspace(rstar-5*delta, rstar+5*delta, num=1000)

s3val = s3f(rarray)
s5val = s5f(rarray)




xtick_values = [rstar-5*delta, rstar-4*delta, rstar-3*delta, rstar-2*delta, rstar-delta, rstar, rstar+delta, rstar+2*delta, rstar+3*delta, rstar+4*delta, rstar+5*delta]

# xtick_labels = [r'$r^*-4\Delta$', r'$r^*-3\Delta$', r'$r^*-2\Delta$', r'$r^*-\Delta$',
                # r'$r^*$',
                # r'$r^*+\Delta$', r'$r^*+2\Delta$', r'$r^*+3\Delta$', r'$r^*+4\Delta$']
xtick_labels = ['-10', '-8', '-6', '-4', '-2', '0', '2', '4', '6', '8', '10']

fig, ax = plt.subplots(1,1, figsize=(8,8))
plot3 = ax.plot(rarray, s3val, label='Polynomial order 3')
plot5 = ax.plot(rarray, s5val, label='Polynomial order 5')
for i in range(-10, 11):
    ax.axvline(rstar+i/2*delta, color='black', linestyle='--', alpha=0.5)
ax.set_xticks(xtick_values)
ax.set_xticklabels(xtick_labels)
ax.set_xlabel(f'r = rstar + _ gridboxes (Lx/nx)')
plt.legend()

# plot = spb.plot_piecewise((smooth3, (r, rstar-5*delta, rstar+5*delta)), (smooth5, (r, rstar-5*delta, rstar+5*delta)), show=False, label=['3', '5'], legend='True')
plt.savefig(f'/data/home/sh1293/firedrake-real-opt_jun25/src/gusto/examples/shallow_water/smooth_pv_comp.pdf')
print(f'Plot made:\n /data/home/sh1293/firedrake-real-opt_jun25/src/gusto/examples/shallow_water/smooth_pv_comp.pdf')
