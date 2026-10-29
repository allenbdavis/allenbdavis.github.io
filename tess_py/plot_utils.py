"""Shared Matplotlib helpers: dark-theme palette, axis styling, PNG -> base64.

Pure Matplotlib (Agg backend) -- no Pyodide-specific code, runs anywhere.
"""
import io, base64
import matplotlib
matplotlib.use('Agg')            # must precede any pyplot import
import matplotlib.pyplot as plt

# Shared colour palette
PALETTE = dict(
    fig_bg   = '#0a0e1a',
    ax_bg    = '#05080f',
    line     = '#4af3c8',
    ax_col   = '#ffffff',
    title    = '#ffffff',
    grid     = '#1a2340',
    fit      = '#f3c84a',
    odd_col  = '#f35a9a',
    even_col = '#4af3c8',
)


S = PALETTE    # short alias used by the plotting modules

def style(ax, title='', xlabel='Time (days)', ylabel='Relative Flux'):
    ax.set_facecolor(S['ax_bg'])
    ax.set_xlabel(xlabel, color=S['ax_col'], fontsize=8, labelpad=4)
    ax.set_ylabel(ylabel, color=S['ax_col'], fontsize=8, labelpad=4)
    if title:
        ax.set_title(title, color=S['title'], fontsize=8, pad=5)
    for sp in ax.spines.values():
        sp.set_edgecolor(S['grid'])
    ax.tick_params(colors=S['ax_col'], labelsize=7)
    ax.grid(True, color=S['grid'], lw=0.4, alpha=0.8)

def to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()
