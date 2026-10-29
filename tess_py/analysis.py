"""Pure-numpy light-curve analysis: phase folding, vectorised BLS box fit,
iterative period search, and the box-depth refinement used by the plots.
No Matplotlib, no JSON, no global state.
"""
import numpy as np


def fold_phase(t, t0, period):
    """Phase in [-0.5, 0.5), transits at phase 0."""
    ph = ((t - t0) % period) / period
    return np.where(ph > 0.5, ph - 1.0, ph)


def bls_vectorized(ph, flux, sigma, n_centers=60, n_widths=15):
    """
    Vectorized BLS-style box fit on already-folded data.

    Avoids all Python loops — uses numpy broadcasting to evaluate
    n_centers x n_widths box models simultaneously.

    Depth = mean(out) - mean(in), computed analytically per box.
    Returns: (best_center, best_w, best_depth, best_chi2, n_in)
    """
    N       = len(flux)
    var     = sigma ** 2
    S_total = float(np.sum(flux))          # sum of all flux values

    centers = np.linspace(-0.40, 0.40, n_centers)   # (C,)
    widths  = np.linspace(0.01,  0.15,  n_widths)    # (W,)

    # ph: (N,)  centers: (C,)  -> dist: (C, N)
    dist = np.abs(ph[np.newaxis, :] - centers[:, np.newaxis])  # (C, N)

    best_chi2   = np.inf
    best_center = 0.0
    best_w      = 0.05
    best_depth  = 0.0

    for w in widths:
        in_box = dist < w                          # (C, N) bool
        n_in   = in_box.sum(axis=1)                # (C,)
        n_out  = N - n_in

        # Skip boxes with too few points
        valid = (n_in >= 3) & (n_out >= 10)
        if not valid.any():
            continue

        S_in  = (in_box * flux[np.newaxis, :]).sum(axis=1)   # (C,)
        S_out = S_total - S_in

        mean_in  = np.where(n_in  > 0, S_in  / n_in,  1.0)
        mean_out = np.where(n_out > 0, S_out / n_out,  1.0)
        dep      = mean_out - mean_in                          # (C,)

        # Only consider positive depths
        valid = valid & (dep > 0)
        if not valid.any():
            continue

        # chi2: sum of squared residuals / var
        # = (sum(flux^2) - n_out*mean_out^2 - n_in*(mean_out-dep)^2) / var  [simplified]
        # Direct vectorized form:
        model_mean = np.where(in_box,
                              (mean_out - dep)[: , np.newaxis],
                               mean_out[:, np.newaxis])        # (C, N)
        chi2 = np.sum((flux[np.newaxis, :] - model_mean) ** 2, axis=1) / var  # (C,)

        # Mask invalid entries
        chi2 = np.where(valid, chi2, np.inf)
        idx  = int(np.argmin(chi2))

        if chi2[idx] < best_chi2:
            best_chi2   = float(chi2[idx])
            best_center = float(centers[idx])
            best_w      = float(w)
            best_depth  = float(dep[idx])

    mask_best = np.abs(ph - best_center) < best_w
    n_in      = int(mask_best.sum())
    return best_center, best_w, best_depth, best_chi2, n_in


def refine_box(ph, flux, c0, w, dep, fallback='median'):
    """
    Re-measure depth with the inner 60% of the box against the outer region,
    avoiding edge dilution.  Returns (plot_base, plot_depth).
    `fallback` is how the baseline is estimated when too few points are
    available ('median' for fixed-period folds, 'mean' for the period search).
    """
    mask_best = np.abs(ph - c0) < w
    mask_core = np.abs(ph - c0) < w * 0.6
    mask_out  = np.abs(ph - c0) > w * 1.5
    if mask_core.sum() >= 3 and mask_out.sum() >= 10:
        plot_depth = float(np.median(flux[mask_out])) - float(np.median(flux[mask_core]))
        plot_base  = float(np.median(flux[mask_out]))
    else:
        plot_depth = dep
        agg        = np.median if fallback == 'median' else np.mean
        plot_base  = float(agg(flux[~mask_best]))
    return plot_base, plot_depth


def search_period(t, flux, sigma, t0_guess, period_guess, n_periods=200):
    """
    Iterative-zoom grid search over trial periods (+/-10% of the guess, then
    narrowing to 0.0001 d), minimising the BLS chi-squared at each period.
    Returns a dict with the best period and fold, plus summary statistics.
    """
    t0_guess     = float(t0_guess)
    period_guess = float(period_guess)

    p_lo, p_hi = 0.9 * period_guess, 1.1 * period_guess
    best_chi2   = np.inf
    best_period = period_guess
    best_center = 0.0
    best_w      = 0.05
    best_depth  = 0.0

    while True:
        periods = np.linspace(p_lo, p_hi, int(n_periods))
        step    = periods[1] - periods[0]

        for P in periods:
            ph = fold_phase(t, t0_guess, P)
            c, w, dep, chi2, _ = bls_vectorized(ph, flux, sigma)
            if chi2 < best_chi2:
                best_chi2   = chi2
                best_period = float(P)
                best_center = c
                best_w      = w
                best_depth  = dep

        if step < 0.0001:           # converged
            break

        p_lo = max(best_period - step, 0.5 * period_guess)
        p_hi = min(best_period + step, 1.5 * period_guess)
        best_chi2 = np.inf          # re-evaluate every point in the new window

    ph_best   = fold_phase(t, t0_guess, best_period)
    mask_best = np.abs(ph_best - best_center) < best_w
    n_in      = int(mask_best.sum())
    return dict(
        period = best_period, ph = ph_best, center = best_center, width = best_w,
        depth  = best_depth, chi2 = best_chi2, n_in = n_in,
        chi2r  = best_chi2 / max(len(flux) - 2, 1),
        snr    = best_depth / (sigma / max(n_in ** 0.5, 1)),
    )
