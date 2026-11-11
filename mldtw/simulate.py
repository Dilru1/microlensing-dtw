"""Toy Roman-like light-curve simulator for exploratory analysis.

Fixed 10-day window sampled every 30 min (480 epochs), flux normalised to the
baseline, constant 1% Gaussian photometric noise. Five classes:
0 constant, 1 variable star, 2 stellar microlensing (point lens),
3 free-floating-planet microlensing (finite source), 4 stellar flare.
"""
import numpy as np
import VBMicrolensing

_VBM = VBMicrolensing.VBMicrolensing()

T_SPAN, DT = 10.0, 1.0 / 48.0          # days, 30-minute cadence
T = np.arange(0.0, T_SPAN, DT)          # 480 epochs
SIGMA = 0.01                            # relative photometric noise
CLASSES = ["constant", "variable", "stellar event", "FFP event", "flare"]


def pspl_mag(u):
    return (u**2 + 2) / (u * np.sqrt(u**2 + 4))


def fs_mag(t, t0, tE, u0, rho):
    """Uniform finite-source point-lens magnification (VBMicrolensing, Bozza et al.)."""
    u = np.sqrt(u0**2 + ((t - t0) / tE) ** 2)
    return np.array([_VBM.ESPLMag2(float(ui), float(rho)) for ui in u])


def blend(A, fs):
    return fs * A + (1 - fs)


def stellar_event(rng):
    p = dict(t0=rng.uniform(3, 7), tE=10 ** rng.uniform(np.log10(3), np.log10(60)),
             u0=rng.uniform(0.02, 0.6), fs=rng.uniform(0.3, 1.0))
    u = np.sqrt(p["u0"] ** 2 + ((T - p["t0"]) / p["tE"]) ** 2)
    return blend(pspl_mag(u), p["fs"]), p


def ffp_event(rng):
    rho = 10 ** rng.uniform(-1, np.log10(3))
    p = dict(t0=rng.uniform(3, 7), tE=10 ** rng.uniform(np.log10(0.05), np.log10(0.5)),
             u0=rng.uniform(0, max(0.5, 0.8 * rho)), rho=rho, fs=rng.uniform(0.5, 1.0))
    return blend(fs_mag(T, p["t0"], p["tE"], p["u0"], rho), p["fs"]), p


def flare(rng):
    """Davenport (2014)-like template: fast polynomial rise, two-exponential decay."""
    t_peak, t_half = rng.uniform(3, 7), 10 ** rng.uniform(-2, np.log10(0.2))
    amp = 10 ** rng.uniform(np.log10(0.05), 0)
    x = (T - t_peak) / t_half
    f = np.zeros_like(T)
    rise = (x > -1) & (x <= 0)
    f[rise] = 1 + 1.941 * x[rise] - 0.175 * x[rise] ** 2 - 2.246 * x[rise] ** 3 - 1.125 * x[rise] ** 4
    dec = x > 0
    f[dec] = 0.6890 * np.exp(-1.600 * x[dec]) + 0.3030 * np.exp(-0.2783 * x[dec])
    return 1 + amp * np.clip(f, 0, None), dict(t0=t_peak, t_half=t_half, amp=amp)


def variable(rng):
    P, amp, ph = 10 ** rng.uniform(np.log10(0.2), np.log10(5)), 10 ** rng.uniform(-1.5, -0.5), rng.uniform(0, 2 * np.pi)
    w = 2 * np.pi * T / P + ph
    return 1 + amp * (np.sin(w) + 0.3 * np.sin(2 * w) + 0.1 * np.sin(3 * w)), dict(P=P, amp=amp)


def constant(rng):
    return np.ones_like(T), {}


GENERATORS = [constant, variable, stellar_event, ffp_event, flare]


def make_dataset(n_per_class, seed=0, classes=range(5)):
    rng = np.random.default_rng(seed)
    X, y, params = [], [], []
    for k in classes:
        made = 0
        while made < n_per_class:
            f, p = GENERATORS[k](rng)
            # keep only events with a detectable signal (peak > 5 sigma)
            if k in (2, 3, 4) and f.max() - 1 < 5 * SIGMA:
                continue
            X.append(f + rng.normal(0, SIGMA, T.size)); y.append(k); params.append(p)
            made += 1
    return np.array(X), np.array(y), params
