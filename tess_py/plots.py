"""Matplotlib figures for the Phase 1 light-curve tools.
Each function takes plain arrays and returns a Figure (callers encode it).
"""
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import S, style


def box_outline(c0, w, base, depth):
    """x, y of the yellow box-model overlay."""
    bx = [c0-w-.001, c0-w, c0-w,         c0+w,         c0+w, c0+w+.001]
    by = [base,      base, base-depth,   base-depth,   base, base]
    return bx, by


def lightcurve_fig(t, flux, title):
    fig, ax = plt.subplots(figsize=(7.5, 2.6), facecolor=S['fig_bg'])
    ax.plot(t, flux, color=S['line'], lw=0.55, alpha=0.9)
    style(ax, title=title)
    fig.tight_layout(pad=0.8)
    return fig, ax


def fold_fig(ph, flux, box, period, snr):
    """box = (c0, w, base, depth) for the fitted box overlay."""
    sidx = np.argsort(ph)
    fig, ax = plt.subplots(figsize=(5.2, 2.4), facecolor=S['fig_bg'])
    ax.scatter(ph[sidx], flux[sidx], s=1.2, c=S['line'], alpha=0.45, linewidths=0)
    bx, by = box_outline(*box)
    ax.plot(bx, by, color=S['fit'], lw=1.5, alpha=0.9, zorder=5)
    style(ax, title=f'Phase Fold   P = {period:.4f} d   SNR = {snr:.1f}',
          xlabel='Phase', ylabel='Relative Flux')
    ax.set_xlim(-0.5, 0.5)
    fig.tight_layout(pad=0.8)
    return fig


def oddeven_fig(ph, flux, even):
    fig, ax = plt.subplots(figsize=(5.2, 2.4), facecolor=S['fig_bg'])
    for mask, col, lbl in [(even, S['even_col'], 'Even'), (~even, S['odd_col'], 'Odd')]:
        idx = np.argsort(ph[mask])
        ax.scatter(ph[mask][idx], flux[mask][idx],
                   s=1.2, c=col, alpha=0.5, linewidths=0, label=lbl)
    ax.legend(fontsize=7, facecolor=S['fig_bg'], labelcolor=S['title'],
              loc='lower right', markerscale=4, framealpha=0.7)
    style(ax, title='Odd / Even Transit Comparison',
          xlabel='Phase', ylabel='Relative Flux')
    ax.set_xlim(-0.5, 0.5)
    fig.tight_layout(pad=0.8)
    return fig
