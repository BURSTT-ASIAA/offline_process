import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from matplotlib.ticker import AutoMinorLocator

# common title

FreqTitle = "Frequency (MHz)"

TimeTitle = "Time"
UTCTitle = "Time (UTC)"
PhaseTitle = "Phase (rad)"

PowerTitle_dB = "Power (dB)"
PowerTitle_dBm = "Power (dBm)"
PSDTitle = "Power spectral density (dBm/MHz)"

def get_beam_colors(nAnt, darken_factor=0.8, cmap_name='gist_rainbow'):
    """
    Generates a list of darkened colors for nAnt antennas in a single call.
    """
    #turbo gist_rainbow nipy_spectral, rainbow: green is too light, need to be darkened

    base_cmap = plt.get_cmap(cmap_name)
    # Calculate rainbow color based on beam index
    indices = np.linspace(0, 1, nAnt)
    colors = base_cmap(indices)
    
    # Darken RGB components (first 3 columns), keep Alpha (last column)
    colors[:, :3] *= darken_factor
    #dark_c = (orig_c[0]*0.8, orig_c[1]*0.8, orig_c[2]*0.8, 1.0)
    return colors


def set_style(target=None):
    """
    Applies astrophysics publication styling.
    
    If `target` is None, configures global `plt.rcParams` for all future plots.
    If `target` is an existing Figure, Axes, or sequence of Axes, styles the
    existing objects directly (ticks inward, mirrored spines, font sizes, etc.).

    Parameters
    ----------
    target : plt.Figure, plt.Axes, sequence of plt.Axes, or None
        The target matplotlib object(s) to style retroactively.
    """
    style_params = {
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9.5,
        'ytick.labelsize': 9.5,
        'legend.fontsize': 9,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'xtick.major.size': 5.0,
        'xtick.minor.size': 2.5,
        'ytick.major.size': 5.0,
        'ytick.minor.size': 2.5,
        'xtick.major.width': 1.0,
        'xtick.minor.width': 0.7,
        'ytick.major.width': 1.0,
        'ytick.minor.width': 0.7,
        'axes.linewidth': 1.0,
        'legend.frameon': True,
        'legend.framealpha': 0.85,
        'legend.edgecolor': '0.8',
        'legend.fancybox': False,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    }

    # Update global settings
    plt.rcParams.update(style_params)

    if target is None:
        return

    # Extract axes list depending on input type
    if isinstance(target, plt.Figure):
        axes_list = target.get_axes()
    elif isinstance(target, plt.Axes):
        axes_list = [target]
    elif hasattr(target, '__iter__'):
        axes_list = list(target)
    else:
        raise TypeError(f"Unsupported target type: {type(target)}")

    for ax in axes_list:
        # Enable minor ticks and mirrored inward ticks
        ax.minorticks_on()
        ax.tick_params(
            axis='both',
            which='both',
            direction='in',
            top=True,
            right=True
        )
        ax.tick_params(
            axis='both',
            which='major',
            length=5.0,
            width=1.0,
            labelsize=9.5
        )
        ax.tick_params(
            axis='both',
            which='minor',
            length=2.5,
            width=0.7
        )

        # Update spine linewidths
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

        # Update label and title sizes
        ax.xaxis.label.set_size(11)
        ax.yaxis.label.set_size(11)
        ax.title.set_size(12)

        # Update font to serif if text elements already exist
        for item in ([ax.xaxis.label, ax.yaxis.label, ax.title] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_family('serif')

        # Update legend styling if already present
        leg = ax.get_legend()
        if leg is not None:
            leg.set_frame_on(True)
            leg.get_frame().set_alpha(0.85)
            leg.get_frame().set_edgecolor('0.8')
            for text in leg.get_texts():
                text.set_fontsize(9)
                text.set_family('serif')

    # Redraw canvas if applying to an existing figure
    if isinstance(target, plt.Figure):
        target.canvas.draw_idle()
    elif len(axes_list) > 0 and axes_list[0].figure is not None:
        axes_list[0].figure.canvas.draw_idle()

def create_astro_figure(width='single', aspect_ratio=1.0/1.3, nrows=1, ncols=1, sharex=False, sharey=False):
    """
    Generates a publication-ready figure and axis/axes object.

    Parameters
    ----------
    width : float or str
        'single' (~3.54 in / 90 mm) for 1-column, 
        'double' (~7.1 in / 180 mm) for 2-column, or custom width in inches.
    aspect_ratio : float
        Height / Width ratio. Default is ~0.77 (standard for astrophysics subplots).
    """
    set_style()

    if width == 'single':
        fig_w = 3.54
    elif width == 'double':
        fig_w = 7.08
    else:
        fig_w = float(width)

    fig_h = fig_w * aspect_ratio * (nrows / ncols)

    # Use constrained_layout=True to guarantee labels are never pushed off canvas
    fig, axes = plt.subplots(
        nrows=nrows, 
        ncols=ncols, 
        figsize=(fig_w, fig_h), 
        sharex=sharex, 
        sharey=sharey,
        constrained_layout=True
    )

    # Helper function to activate minor ticks on all sub-axes
    ax_list = [axes] if not hasattr(axes, '__iter__') else axes.flat
    for ax in ax_list:
        ax.minorticks_on()
        ax.tick_params(which='both', direction='in', top=True, right=True)

    return fig, axes