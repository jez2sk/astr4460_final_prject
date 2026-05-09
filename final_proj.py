import numpy as np
import matplotlib.pyplot as plt
import athena_mc as athenamc
from scipy.optimize import curve_fit
from scipy.integrate import quad

def power_law(E, A, alpha):
    return A * E**(-alpha)

def pow(x, y, yerr):

    popt, pcov = curve_fit(power_law, x, y, sigma=yerr, absolute_sigma=True, p0=[1e25, 2])

    A_fit, alpha_fit = popt
    A_err, alpha_err = np.sqrt(np.diag(pcov))

    print("power law fit parameters:")
    print(f"A = {A_fit:.3e} ± {A_err:.3e}")
    print(f"alpha = {alpha_fit:.3f} ± {alpha_err:.3f}")
    print()

    return A_fit, A_err, alpha_fit, alpha_err

def blackbody(E, A, T):
    return A * (E**3) / (np.exp(E / T) - 1)

def bb(x, y, yerr):

    popt, pcov = curve_fit(blackbody, x, y, sigma=yerr, absolute_sigma=True, p0=[1e10, 100])
    
    A_fit, T_fit = popt
    A_err, T_err = np.sqrt(np.diag(pcov))

    print("bb fit parameters:")
    print(f"A = {A_fit:.3e} ± {A_err:.3e}")
    print(f"T = {T_fit:.3f} ± {T_err:.3f}")
    print()

    return A_fit, A_err, T_fit, T_err

def diskbb_model(E, A, Tin, Tout):

    def integrand(T, E):
        return (T / Tin)**(-11/3) * blackbody(E, 1, T) / Tin

    E = np.atleast_1d(E)
    vals = []

    for Ei in E:
        integral, _ = quad(integrand, Tout, Tin, args=(Ei,))
        vals.append(A * integral)

    return np.array(vals)

def diskbb(x, y, yerr):

    popt, pcov = curve_fit(diskbb_model, x, y, sigma=yerr, absolute_sigma=True, p0=[1e20, 500, 50], maxfev=20000)

    A_fit, Tin_fit, Tout_fit = popt
    A_err, Tin_err, Tout_err = np.sqrt(np.diag(pcov))

    print("diskbb fit parameters:")
    print(f"A     = {A_fit:.3e} ± {A_err:.3e}")
    print(f"Tin   = {Tin_fit:.3f} ± {Tin_err:.3f}")
    print(f"Tout  = {Tout_fit:.3f} ± {Tout_err:.3f}")
    print()

    return (A_fit, A_err, Tin_fit, Tin_err, Tout_fit, Tout_err)

def comp(E, A_pow, alpha, A_bb, Tin, Tout):
    return power_law(E, A_pow, alpha) + diskbb_model(E, A_bb, Tin, Tout)

def composite(x, y, yerr):

    popt, pcov = curve_fit(comp, x, y, sigma=yerr, absolute_sigma=True, p0=[1e22, 2, 1e20, 500, 50], maxfev=20000)

    A_pow_fit, alpha_fit, A_bb_fit, Tin_fit, Tout_fit = popt
    A_pow_err, alpha_err, A_bb_err, Tin_err, Tout_err = np.sqrt(np.diag(pcov))

    print("Composite fit parameters:")
    print(f"A_pow  = {A_pow_fit:.3e} ± {A_pow_err:.3e}")
    print(f"alpha  = {alpha_fit:.3f} ± {alpha_err:.3f}")
    print(f"A_bb   = {A_bb_fit:.3e} ± {A_bb_err:.3e}")
    print(f"Tin    = {Tin_fit:.3f} ± {Tin_err:.3f}")
    print(f"Tout   = {Tout_fit:.3f} ± {Tout_err:.3f}")
    print()

    return (A_pow_fit, A_pow_err, alpha_fit, alpha_err, A_bb_fit, A_bb_err, Tin_fit, Tin_err, Tout_fit, Tout_err)

def comp_bb(E, A_pow, alpha, A_bb, T):
    return power_law(E, A_pow, alpha) + blackbody(E, A_bb, T)

def composite_bb(x, y, yerr):

    popt, pcov = curve_fit(comp_bb, x, y, sigma=yerr, absolute_sigma=True, p0=[1e22, 2, 1e20, 500], maxfev=20000)

    A_pow_fit, alpha_fit, A_bb_fit, T_fit = popt
    A_pow_err, alpha_err, A_bb_err, T_err = np.sqrt(np.diag(pcov))

    print("Composite fit parameters:")
    print(f"A_pow  = {A_pow_fit:.3e} ± {A_pow_err:.3e}")
    print(f"alpha  = {alpha_fit:.3f} ± {alpha_err:.3f}")
    print(f"A_bb   = {A_bb_fit:.3e} ± {A_bb_err:.3e}")
    print(f"Tin    = {T_fit:.3f} ± {T_err:.3f}")
    print()

    return (A_pow_fit, A_pow_err, alpha_fit, alpha_err, A_bb_fit, A_bb_err, T_fit, T_err)