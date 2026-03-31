import fcsparser
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica']
mpl.rcParams['axes.titleweight'] = 'bold'

# --- Font sizes for panel figures ---
FONTSIZE_TITLE  = 20   # subplot titles
FONTSIZE_LABEL  = 18   # axis labels (xlabel / ylabel)
FONTSIZE_TICK   = 18   # tick labels
FONTSIZE_LEGEND = 15   # legend text
FONTSIZE_CORR   = 20   # numbers in correlation matrices
FONTSIZE_CBAR       = 18   # colorbar tick labels
FONTSIZE_CBAR_LABEL = 20   # colorbar axis labels ('Count (log)', 'Correlation')

# File path
fichier_analyse = "all_data/flow_cytometry_data.fcs"

# Channel names
canal_chloro = "Red Fluorescence (RED-HLin)"
canal_fsc = "Forward Scatter (FSC-HLin)"
canal_ssc = "Side Scatter (SSC-HLin)"


def analyser_panel_datasets(fichier, canal_chloro, canal_fsc, canal_ssc,
                            indices_datasets=None, facteurs_dilution=None):
    """
    Creates a panel figure combining distributions, scatter plots and correlation
    matrices for all datasets, plus a comparative box plot summary.
    Histogram counts are corrected by dilution factors.

    Layout (GridSpec 11 rows x 4 columns):
    - For each dataset (2 rows x 3 columns + correlation spanning 2 rows, col 4):
        Row 1: Chlorophyll dist | FSC dist | SSC dist | Correlation matrix (rowspan=2)
        Row 2: FSC vs SSC      | FSC vs Chloro | SSC vs Chloro |
    - Last row: Comparative box plots across all datasets (colspan=4)

    Parameters:
    - fichier: path to the FCS file
    - canal_chloro, canal_fsc, canal_ssc: channel names
    - indices_datasets: list of dataset indices (default [0, 1, 2, 3, 6])
    - facteurs_dilution: inverse dilution factors (default [100, 100, 100, 10, 10])
    """
    if indices_datasets is None:
        indices_datasets = [0, 1, 2, 3, 6]
    if facteurs_dilution is None:
        facteurs_dilution = [100, 100, 100, 10, 10]

    couleurs = ['mediumseagreen', 'grey', 'tomato', 'teal', 'orange']
    conc_labels = [r'$C_0 = 1$', r'$C_0 = 1/2$', r'$C_0 = 1/4$',
                   r'$C_0 = 1/8$', r'$C_0 = 1/16$']
    nom = Path(fichier).stem

    # Load all datasets
    datasets = []
    labels = []
    for i in indices_datasets:
        meta, data = fcsparser.parse(fichier, reformat_meta=True, data_set=i)
        datasets.append(data)
        sample_id = meta.get('GTI$SAMPLEID', f'Dataset {i}')
        labels.append(f"Dataset {i} - {sample_id}")

    n_ds = len(datasets)

    # Compute global x-axis limits (positive values only for log scale)
    def _pos_minmax(data_list, canal):
        vals = np.concatenate([d[canal][d[canal] > 0].values for d in data_list])
        return vals.min(), vals.max()

    xlim_chloro = _pos_minmax(datasets, canal_chloro)
    xlim_fsc    = _pos_minmax(datasets, canal_fsc)
    xlim_ssc    = _pos_minmax(datasets, canal_ssc)

    # Common logarithmic bins per channel (101 edges = 100 bins)
    bins_chloro = np.logspace(np.log10(xlim_chloro[0]), np.log10(xlim_chloro[1]), 101)
    bins_fsc    = np.logspace(np.log10(xlim_fsc[0]),    np.log10(xlim_fsc[1]),    101)
    bins_ssc    = np.logspace(np.log10(xlim_ssc[0]),    np.log10(xlim_ssc[1]),    101)

    fig = plt.figure(figsize=(22, 3 * 2 * n_ds + 5))
    fig.patch.set_facecolor('white')

    from matplotlib.gridspec import GridSpecFromSubplotSpec

    h_gap = 0.2  # vertical gap between distribution and density rows (in row-height units)

    # Outer GridSpec: 4 rows x 2 cols
    gs_outer = fig.add_gridspec(
        4, 2,
        height_ratios=[2 * n_ds + h_gap, 0.12, 0.15, 1.8],
        width_ratios=[3, 0.8],
        hspace=0.07, wspace=0.15)

    # Left sub-grid: distribution rows (n_ds) + spacer + density rows (n_ds)
    gs_left = GridSpecFromSubplotSpec(
        2 * n_ds + 1, 3,
        subplot_spec=gs_outer[0, 0],
        height_ratios=[1] * n_ds + [h_gap] + [1] * n_ds,
        hspace=0.25, wspace=0.25)

    # Right sub-grid: n_ds equal rows for correlation matrices
    gs_right = GridSpecFromSubplotSpec(
        n_ds, 1,
        subplot_spec=gs_outer[0, 1],
        hspace=0.0)

    # Axis lists by type for shared x-limits and common colorbars
    axes_chloro, axes_fsc_dist, axes_ssc_dist = [], [], []
    axes_fsc_ssc, axes_fsc_chloro, axes_ssc_chloro = [], [], []
    hbs_fsc_ssc, hbs_fsc_chloro, hbs_ssc_chloro = [], [], []
    axes_corr = []
    last_im = None

    for idx, (data, label, couleur, facteur) in enumerate(
            zip(datasets, labels, couleurs, facteurs_dilution)):
        row_dist = idx
        row_dens = n_ds + 1 + idx  # +1 for the spacer row
        corr_matrix = data[[canal_chloro, canal_fsc, canal_ssc]].corr()

        conc_label = conc_labels[idx]

        legend_loc = 'upper left' if idx == 2 else 'upper right'

        def _hist_corr(ax, values, bins, titre, show_ylabel=False, show_legend=False,
                       _conc_label=conc_label, _couleur=couleur, _facteur=facteur,
                       _legend_loc=legend_loc):
            from matplotlib.patches import Patch
            values = values[values > 0]
            counts, bin_edges = np.histogram(values, bins=bins)
            ax.bar(bin_edges[:-1], counts * _facteur,
                   width=np.diff(bin_edges), align='edge',
                   alpha=0.7, color=_couleur, edgecolor=_couleur)
            ax.set_xscale('log')
            if show_ylabel:
                ax.set_ylabel('Count', fontsize=FONTSIZE_LABEL)
            ax.set_title(titre, fontweight='bold', fontsize=FONTSIZE_TITLE)
            ax.tick_params(labelsize=FONTSIZE_TICK)
            ax.set_facecolor('white')
            ax.grid(False)
            if show_legend:
                ax.legend(handles=[Patch(facecolor=_couleur, alpha=0.7, label=_conc_label)],
                          loc=_legend_loc, fontsize=FONTSIZE_LEGEND, frameon=False)

        # --- Distribution row (gs_left) ---
        ax0 = fig.add_subplot(gs_left[row_dist, 0])
        _hist_corr(ax0, data[canal_chloro], bins_chloro, 'Chl' if idx == 0 else '', show_ylabel=True)
        axes_chloro.append(ax0)

        ax1 = fig.add_subplot(gs_left[row_dist, 1])
        _hist_corr(ax1, data[canal_fsc], bins_fsc, 'FSC' if idx == 0 else '')
        axes_fsc_dist.append(ax1)

        ax2 = fig.add_subplot(gs_left[row_dist, 2])
        _hist_corr(ax2, data[canal_ssc], bins_ssc, 'SSC' if idx == 0 else '', show_legend=True)
        axes_ssc_dist.append(ax2)

        # --- Density plot row (gs_left) ---
        ax = fig.add_subplot(gs_left[row_dens, 0])
        hb = ax.hexbin(data[canal_fsc], data[canal_ssc], gridsize=50,
                       cmap='viridis', mincnt=1, bins='log')
        ax.set_title('SSC(FSC)' if idx == 0 else '', fontweight='bold', fontsize=FONTSIZE_TITLE)
        ax.tick_params(labelsize=FONTSIZE_TICK)
        ax.grid(False)
        axes_fsc_ssc.append(ax)
        hbs_fsc_ssc.append(hb)

        ax = fig.add_subplot(gs_left[row_dens, 1])
        hb = ax.hexbin(data[canal_fsc], data[canal_chloro], gridsize=50,
                       cmap='inferno', mincnt=1, bins='log')
        ax.set_title('Chl(FSC)' if idx == 0 else '', fontweight='bold', fontsize=FONTSIZE_TITLE)
        ax.tick_params(labelsize=FONTSIZE_TICK)
        ax.grid(False)
        axes_fsc_chloro.append(ax)
        hbs_fsc_chloro.append(hb)

        ax = fig.add_subplot(gs_left[row_dens, 2])
        hb = ax.hexbin(data[canal_ssc], data[canal_chloro], gridsize=50,
                       cmap='YlOrRd_r', mincnt=1, bins='log')
        ax.set_title('Chl(SSC)' if idx == 0 else '', fontweight='bold', fontsize=FONTSIZE_TITLE)
        ax.tick_params(labelsize=FONTSIZE_TICK)
        ax.grid(False)
        axes_ssc_chloro.append(ax)
        hbs_ssc_chloro.append(hb)

        # --- Correlation matrix (gs_right, one row per dataset, equal height) ---
        ax = fig.add_subplot(gs_right[idx, 0])
        last_im = ax.imshow(corr_matrix, cmap='BrBG', vmin=0, vmax=1, aspect='auto')
        labels_corr = ['Chl', 'FSC', 'SSC']
        ax.set_xticks(range(3))
        ax.set_xticklabels(labels_corr)
        ax.set_yticks(range(3))
        ax.set_yticklabels(labels_corr)
        ax.tick_params(labelsize=FONTSIZE_TICK)
        for r in range(3):
            for c in range(3):
                color = 'white' if r == c else 'black'
                ax.text(c, r, f'{corr_matrix.iloc[r, c]:.2f}',
                        ha='center', va='center', fontsize=FONTSIZE_CORR,
                        fontweight='bold', color=color)
        axes_corr.append(ax)

    # Set common color scale per density plot column
    def _set_common_clim(hbs):
        vmin = min(hb.get_array().min() for hb in hbs)
        vmax = max(hb.get_array().max() for hb in hbs)
        for hb in hbs:
            hb.set_clim(vmin, vmax)

    _set_common_clim(hbs_fsc_ssc)
    _set_common_clim(hbs_fsc_chloro)
    _set_common_clim(hbs_ssc_chloro)

    # Apply shared x-axis limits
    for ax in axes_chloro:
        ax.set_xlim(xlim_chloro)
    for ax in axes_fsc_dist:
        ax.set_xlim(xlim_fsc)
    for ax in axes_ssc_dist:
        ax.set_xlim(xlim_ssc)
    for ax in axes_fsc_ssc:
        ax.set_xlim(xlim_fsc)
    for ax in axes_fsc_chloro:
        ax.set_xlim(xlim_fsc)
    for ax in axes_ssc_chloro:
        ax.set_xlim(xlim_ssc)

    # --- Horizontal colorbars (colorbar row of gs_outer) ---
    gs_cbar_left = GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_outer[1, 0], wspace=0.25)
    for col_idx, hbs in enumerate([hbs_fsc_ssc, hbs_fsc_chloro, hbs_ssc_chloro]):
        cax = fig.add_subplot(gs_cbar_left[0, col_idx])
        cb = fig.colorbar(hbs[0], cax=cax, orientation='horizontal')
        cb.set_label('Count', fontsize=FONTSIZE_CBAR_LABEL)
        cb.ax.tick_params(labelsize=FONTSIZE_CBAR)

    cax_corr = fig.add_subplot(gs_outer[1, 1])
    cb_corr = fig.colorbar(last_im, cax=cax_corr, orientation='horizontal')
    cb_corr.set_label('Correlation', fontsize=FONTSIZE_CBAR_LABEL)
    cb_corr.ax.tick_params(labelsize=FONTSIZE_CBAR)

    # --- Bottom row: comparative box plots across all datasets ---
    ax = fig.add_subplot(gs_outer[3, :])
    ax.set_facecolor('white')
    canaux = [canal_chloro, canal_fsc, canal_ssc]
    noms_canaux = ['Chl', 'FSC', 'SSC']
    group_width = n_ds + 1

    for j, (canal, nom_canal) in enumerate(zip(canaux, noms_canaux)):
        for i, (data, couleur) in enumerate(zip(datasets, couleurs)):
            pos = j * group_width + i
            bp = ax.boxplot(data[canal], positions=[pos], widths=0.6,
                            patch_artist=True, manage_ticks=False,
                            medianprops=dict(color='black'))
            bp['boxes'][0].set_facecolor(couleur)
            bp['boxes'][0].set_alpha(0.6)
            ax.plot(pos, data[canal].mean(), marker='^', color='black',
                    markersize=6, zorder=5)

    tick_positions = [j * group_width + (n_ds - 1) / 2 for j in range(3)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(noms_canaux)
    ax.set_ylabel('Count', fontsize=FONTSIZE_LABEL)
    ax.set_title('Comparative summary statistics', fontweight='bold', fontsize=FONTSIZE_TITLE)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.set_yscale('log')
    ax.grid(False)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=couleurs[i], alpha=0.6, label=conc_labels[i])
                       for i in range(n_ds)]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=FONTSIZE_LEGEND,
              frameon=False, ncol=n_ds)

    plt.savefig(f'panel_analyse_{nom}.png', dpi=300, bbox_inches='tight')
    print(f"Figure saved: panel_analyse_{nom}.png")


if __name__ == "__main__":
    analyser_panel_datasets(fichier_analyse, canal_chloro, canal_fsc, canal_ssc,
                            indices_datasets=[0, 1, 2, 3, 6],
                            facteurs_dilution=[100, 100, 100, 10, 10])
