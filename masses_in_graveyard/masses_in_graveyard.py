import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.path as mpath
from matplotlib.collections import LineCollection
from scipy.stats import gaussian_kde
import matplotlib.style as mstyle

if not hasattr(mstyle, "core"):
    mstyle.core = mstyle

import scienceplots
plt.style.use(['science'])

# ─── LVK black hole masses (GWTC-4 BBH events) ───────────────────────────

LVK_MASS1 = [11.54, 33.8, 38.1, 10.3, 44.0, 34.2, 33.76, 39.0, 37.5, 54.0, 45.8, 30.0, 12.4, 68.0, 42.4, 52.0, 55.7, 12.4, 40.0, 29.0, 47.0, 11.1, 29.9, 30.4, 63.8, 60.0, 42.0, 70.0, 11.3, 47.0, 19.3, 25.3, 16.4, 7.3, 42.4, 44.4, 10.9, 41.0, 19.5, 33.2, 12.7, 37.1, 36.6, 24.6, 37.3, 35.8, 25.5, 9.02, 46.0, 46.1, 11.5, 34.9, 37.1, 24.7, 38.3, 10.7, 11.6, 10.9, 10.9, 45.0, 39.8, 38.0, 24.4, 11.9, 13.2, 70.0, 40.3, 47.1, 34.4, 28.6, 11.1, 13.2, 18.6, 45.0, 41.6, 36.8, 65.0, 28.5, 34.0, 55.0, 10.6, 52.0, 33.0, 19.5, 14.1, 27.6, 52.0, 14.5, 31.1, 11.4, 65.0, 37.1, 46.8, 24.4, 12.6, 40.5, 31.3, 31.0, 38.6, 19.8, 52.0, 34.6, 38.8, 7.9, 28.7, 59.0, 42.3, 22.5, 53.0, 40.2, 9.3, 11.1, 11.9, 46.0, 47.0, 35.6, 37.6, 35.6, 45.0, 45.0, 137.0, 49.0, 12.9, 43.0, 20.0, 22.8, 11.6, 56.0, 39.8, 19.4, 23.2, 12.2, 61.0, 65.0, 94.0, 34.0, 12.0, 11.6, 20.6, 45.0, 28.8, 84.0, 64.0, 76.0, 34.5, 54.0, 21.7, 35.0, 28.8, 76.0, 39.3, 32.4, 27.3, 60.0, 33.9, 10.6, 42.0, 44.0, 53.0, 62.0, 70.0, 33.7, 69.0, 35.4, 51.0, 32.2, 44.0, 10.3, 12.3, 35.6, 16.6, 32.0, 45.0, 64.0, 29.1, 46.1, 16.3, 89.0, 32.6, 40.0, 10.0, 51.0, 32.5, 8.4, 27.1, 35.4, 48.0, 37.6, 17.4, 64.0, 3.66, 8.17, 38.0, 13.1, 34.2, 60.0, 28.3, 37.8, 19.3, 40.0, 38.9, 87.0, 37.5, 51.0, 24.1, 35.6, 51.0, 37.7, 10.1, 34.5, 42.2, 5.9, 35.6, 49.4, 45.1, 31.1, 12.1, 24.9, 11.7, 27.3, 10.7, 53.0, 12.1, 29.0, 65.0, 10.7, 11.8, 14.2, 66.3, 41.1, 20.8, 8.8, 9.7, 43.8, 32.6, 43.8, 23.7, 31.9, 23.3, 46.2, 37.7, 41.8, 12.5, 38.9, 11.8, 14.2, 36.6, 19.8, 12.1, 74.0, 54.1, 35.1, 58.0, 71.8, 35.6, 98.4, 65.1, 39.2, 40.9, 36.0, 23.2, 41.3, 105.5, 2.1, 42.0, 51.3, 33.7, 27.7, 24.8, 85.0, 38.3, 34.8, 1.46, 30.9, 34.1, 54.7, 10.6, 28.7, 14.2, 24.8, 34.6]

LVK_MASS2 = [9.95, 26.7, 29.6, 6.9, 31.0, 21.3, 32.26, 30.2, 27.4, 36.0, 34.5, 21.1, 7.1, 49.0, 33.5, 33.0, 42.2, 8.1, 23.6, 21.4, 33.0, 6.5, 24.3, 23.0, 20.8, 47.0, 26.0, 24.0, 7.7, 25.0, 14.4, 20.5, 8.0, 5.14, 31.7, 31.0, 8.0, 19.3, 5.96, 25.5, 8.1, 26.8, 26.6, 20.3, 29.9, 21.9, 11.2, 6.99, 34.0, 34.9, 7.8, 9.8, 31.9, 13.4, 29.6, 7.8, 7.5, 7.7, 6.3, 32.0, 27.7, 28.3, 18.3, 7.8, 7.9, 37.0, 26.5, 35.3, 26.4, 21.6, 7.8, 6.7, 11.3, 32.0, 31.6, 29.7, 40.0, 20.0, 26.4, 38.0, 7.7, 32.0, 22.6, 14.3, 8.3, 10.0, 33.0, 8.4, 21.4, 8.0, 39.0, 18.5, 36.7, 16.9, 8.0, 32.2, 8.5, 23.1, 27.7, 14.7, 36.0, 23.8, 25.9, 5.57, 18.1, 33.0, 32.2, 17.2, 35.0, 35.1, 7.31, 8.3, 6.8, 31.0, 29.0, 27.2, 28.5, 28.2, 23.8, 28.0, 101.0, 33.0, 7.4, 29.0, 10.8, 8.2, 7.4, 29.0, 26.4, 12.6, 17.4, 8.6, 42.0, 42.0, 59.0, 20.8, 7.4, 7.3, 14.7, 25.6, 21.2, 50.0, 35.0, 41.0, 24.4, 29.0, 16.6, 27.1, 23.2, 51.0, 29.2, 23.8, 21.4, 37.0, 21.6, 7.1, 30.0, 27.2, 36.0, 34.0, 35.0, 28.2, 42.0, 22.3, 35.0, 22.6, 29.0, 7.9, 7.6, 27.9, 10.6, 12.5, 30.0, 40.0, 22.7, 36.3, 11.5, 50.0, 20.0, 18.0, 6.6, 33.0, 27.0, 5.74, 16.1, 25.2, 31.0, 25.7, 11.0, 44.0, 1.42, 1.45, 11.3, 7.8, 27.7, 24.0, 14.8, 20.0, 14.0, 32.7, 27.9, 61.0, 27.9, 30.0, 2.83, 27.1, 12.3, 27.4, 7.3, 29.0, 32.6, 1.44, 28.3, 37.0, 34.7, 1.17, 7.7, 18.1, 8.4, 19.2, 6.7, 24.0, 8.3, 5.9, 47.0, 7.7, 7.9, 6.9, 26.8, 20.4, 15.5, 5.1, 2.1, 23.3, 24.5, 34.2, 10.4, 25.8, 2.6, 30.6, 27.6, 29.0, 8.0, 30.2, 6.3, 7.5, 19.9, 11.6, 7.9, 39.4, 40.5, 24.0, 35.0, 44.8, 22.2, 57.2, 40.8, 24.0, 28.4, 18.3, 12.5, 28.3, 76.0, 1.3, 32.0, 30.4, 24.2, 9.0, 18.5, 20.0, 29.0, 27.6, 1.27, 24.9, 24.2, 30.2, 7.8, 20.8, 7.5, 13.6, 30.0]
# ─── Colours ──────────────────────────────────────────────────────────────────
BH_COLOR   = "#008FBF"   # teal / cyan — black holes
NS_COLOR   = "#FF8C00"   # orange — neutron stars
BG_COLOR   = "#0a0a0a"
MASS_SCALE = 5.0   # marker area = MASS_SCALE * mass  (area ∝ mass)


def ms(mass):
    """Marker size (area in pt²) proportional to mass."""
    return 3 * mass


def half_circle_markers():
    """Top/bottom half-circle Path markers for split-color mfinal points."""
    t = np.linspace(0, np.pi, 50)
    top_verts = np.vstack([[-1, 0], np.c_[np.cos(t), np.sin(t)], [1, 0]])
    top_half = mpath.Path(top_verts,
                          [mpath.Path.MOVETO] + [mpath.Path.LINETO] * 50 + [mpath.Path.CLOSEPOLY])

    t = np.linspace(np.pi, 2 * np.pi, 50)
    bot_verts = np.vstack([[1, 0], np.c_[np.cos(t), np.sin(t)], [-1, 0]])
    bot_half = mpath.Path(bot_verts,
                          [mpath.Path.MOVETO] + [mpath.Path.LINETO] * 50 + [mpath.Path.CLOSEPOLY])
    return top_half, bot_half


def generate_bbh_params(num_bbh=300):
    """LVK BBH events merged with a KDE-sampled third-generation BBH population."""
    third_gen_mass1 = gaussian_kde(LVK_MASS1).resample(num_bbh, seed=2026)[0]
    q = np.random.uniform(1, 12, num_bbh)
    third_gen_mass2 = third_gen_mass1 / q

    _filter = np.where((third_gen_mass1 > 5) & (third_gen_mass2 > 5))
    third_gen_mass1 = third_gen_mass1[_filter]
    third_gen_mass2 = third_gen_mass2[_filter]

    mass1 = np.concatenate([LVK_MASS1, third_gen_mass1])
    mass2 = np.concatenate([LVK_MASS2, third_gen_mass2])
    mfinal = 0.95 * (mass1 + mass2)
    return mass1, mass2, mfinal


def generate_bns_params(num_bns=300):
    """Third-generation BNS (neutron star - neutron star) population."""
    mass1 = np.random.uniform(1.18, 2.25, num_bns)
    ratio = np.random.uniform(0.8, 0.97, num_bns)
    mass2 = mass1 * ratio

    _filter = (mass1 > 1) & (mass2 > 1)
    mass1 = mass1[_filter]
    mass2 = mass2[_filter]
    mfinal = 0.95 * (mass1 + mass2)
    return mass1, mass2, mfinal


def generate_nsbh_params(num_nsbh=300):
    """Third-generation NSBH (neutron star - black hole) population."""
    ns = np.random.uniform(1.18, 2.25, num_nsbh)
    bh = np.random.uniform(3, 30, num_nsbh)
    mfinal = 0.95 * (ns + bh)
    return ns, bh, mfinal


def arch_order(mass1, mass2):
    """Top half (highest total mass) -> arch, heaviest in the centre.
    Bottom half (lowest total mass) -> scattered randomly along x.
    Returns the x position of every event and the arch width (arch_n)."""
    total_mass = mass1 + mass2
    n = len(total_mass)
    sorted_idx = np.argsort(total_mass)          # ascending: lightest first

    n_half   = int(n*0.7) 
    low_idx  = sorted_idx[:n_half]               # lightest half -> scatter
    high_idx = sorted_idx[n_half:]                # heaviest half -> arch

    x_of = np.empty(n, dtype=float)

    arch_n      = len(high_idx)
    left, right = 0, arch_n - 1
    for k, ev_idx in enumerate(high_idx):
        if k % 2 == 0:
            x_of[ev_idx] = left;  left  += 1
        else:
            x_of[ev_idx] = right; right -= 1

    rng = np.random.default_rng(42)
    x_of[low_idx] = rng.uniform(0, arch_n - 1, size=n_half)
    return x_of, arch_n


def main():
    bbh_mass1, bbh_mass2, bbh_mfinal = generate_bbh_params()
    bns_mass1, bns_mass2, bns_mfinal = generate_bns_params()
    nsbh_ns, nsbh_bh, nsbh_mfinal    = generate_nsbh_params()

    x_of, arch_n = arch_order(bbh_mass1, bbh_mass2)
    n_all = len(bbh_mass1)

    # ─── Plot ─────────────────────────────────────────────────────────────────
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # ── BBH (LVK + third-gen, unified style) ────────────────────────────────
    r, g, b = mcolors.to_rgb(BH_COLOR)

    segs = [[(x_of[i], bbh_mass2[i]), (x_of[i], bbh_mass1[i])] for i in range(n_all)]
    ax.add_collection(LineCollection(segs, colors="white", linewidths=0.6, alpha=0.2, zorder=1))

    for i in range(n_all):
        ax.annotate("", xy=(x_of[i], bbh_mfinal[i]*0.9), xytext=(x_of[i], bbh_mass1[i]),
                    arrowprops=dict(arrowstyle="->", color="white", lw=0.5, alpha=0.5,
                                    shrinkA=0, shrinkB=0),
                    zorder=2)

    s_m2     = [ms(m) for m in bbh_mass2]
    s_m1     = [ms(m) for m in bbh_mass1]
    s_mfinal = [ms(m) for m in bbh_mfinal]
    fc = [(r, g, b, 0.5)] * n_all
    ec = [(r, g, b, 1.0)] * n_all
    ax.scatter(x_of, bbh_mass2,  facecolors=fc, edgecolors=ec, linewidths=0.8, s=s_m2,     zorder=3)
    ax.scatter(x_of, bbh_mass1,  facecolors=fc, edgecolors=ec, linewidths=0.8, s=s_m1,     zorder=3)
    ax.scatter(x_of, bbh_mfinal, facecolors=fc, edgecolors=ec, linewidths=0.8, s=s_mfinal, zorder=3)

    # ── NSBH: third-gen NS-BH systems, scattered randomly along x ───────────
    n_nsbh   = len(nsbh_ns)
    rng_nsbh = np.random.default_rng(77)
    x_nsbh   = rng_nsbh.uniform(0, arch_n - 1, size=n_nsbh)

    r_bh, g_bh, b_bh   = mcolors.to_rgb(BH_COLOR)
    r_ns2, g_ns2, b_ns2 = mcolors.to_rgb(NS_COLOR)

    segs_nsbh = [[(x_nsbh[i], nsbh_ns[i]), (x_nsbh[i], nsbh_bh[i])] for i in range(n_nsbh)]
    ax.add_collection(LineCollection(segs_nsbh, colors="white", linewidths=0.6, alpha=0.2, zorder=1))

    for i in range(n_nsbh):
        ax.annotate("", xy=(x_nsbh[i], nsbh_mfinal[i]), xytext=(x_nsbh[i], nsbh_bh[i]),
                    arrowprops=dict(arrowstyle="->", color="white", lw=0.5, alpha=0.4,
                                    shrinkA=0, shrinkB=0),
                    zorder=2)

    s_nsbh_ns     = [ms(m) for m in nsbh_ns]
    s_nsbh_bh     = [ms(m) for m in nsbh_bh]
    s_nsbh_mfinal = [ms(m) for m in nsbh_mfinal]
    fc_nsbh_ns = [(r_ns2, g_ns2, b_ns2, 0.2)] * n_nsbh
    ec_nsbh_ns = [(r_ns2, g_ns2, b_ns2, 1.0)] * n_nsbh
    fc_nsbh_bh = [(r_bh,  g_bh,  b_bh,  0.2)] * n_nsbh
    ec_nsbh_bh = [(r_bh,  g_bh,  b_bh,  1.0)] * n_nsbh
    ax.scatter(x_nsbh, nsbh_ns,     facecolors=fc_nsbh_ns, edgecolors=ec_nsbh_ns,
               linewidths=0.8, s=s_nsbh_ns,     zorder=3)
    ax.scatter(x_nsbh, nsbh_bh,     facecolors=fc_nsbh_bh, edgecolors=ec_nsbh_bh,
               linewidths=0.8, s=s_nsbh_bh,     zorder=3)
    ax.scatter(x_nsbh, nsbh_mfinal, facecolors=fc_nsbh_bh, edgecolors=ec_nsbh_bh,
               linewidths=0.8, s=s_nsbh_mfinal, zorder=3)

    # ── BNS: third-gen neutron stars, scattered randomly along x ────────────
    n_ns   = len(bns_mass1)
    rng_ns = np.random.default_rng(99)
    x_ns   = rng_ns.uniform(0, arch_n - 1, size=n_ns)

    # Place the 10 lowest-mass2 NS events near the centre
    _ns_low10 = np.argsort(bns_mass2)[:10]
    _center   = (arch_n - 1) / 2
    x_ns[_ns_low10] = rng_ns.uniform(_center - 10, _center + 10, size=10)

    r_ns, g_ns, b_ns = mcolors.to_rgb(NS_COLOR)

    segs_ns = [[(x_ns[i], bns_mass2[i]), (x_ns[i], bns_mass1[i])] for i in range(n_ns)]
    ax.add_collection(LineCollection(segs_ns, colors="white", linewidths=0.6, alpha=0.2, zorder=1))

    s_ns_m2     = [ms(m) for m in bns_mass2]
    s_ns_m1     = [ms(m) for m in bns_mass1]
    s_ns_mfinal = [ms(m) for m in bns_mfinal]
    fc_ns = [(r_ns, g_ns, b_ns, 0.2)] * n_ns
    ec_ns = [(r_ns, g_ns, b_ns, 1.0)] * n_ns
    ax.scatter(x_ns, bns_mass2, facecolors=fc_ns, edgecolors=ec_ns, linewidths=0.8, s=s_ns_m2, zorder=3)
    ax.scatter(x_ns, bns_mass1, facecolors=fc_ns, edgecolors=ec_ns, linewidths=0.8, s=s_ns_m1, zorder=3)

    for i in range(n_ns):
        ax.annotate("", xy=(x_ns[i], bns_mfinal[i]), xytext=(x_ns[i], bns_mass1[i]),
                    arrowprops=dict(arrowstyle="->", color="white", lw=0.5, alpha=0.4,
                                    shrinkA=0, shrinkB=0),
                    zorder=2)

    # mfinal: split marker — top half BH_COLOR, bottom half NS_COLOR
    top_half, bot_half = half_circle_markers()
    fc_fin_bh = [(r_bh, g_bh, b_bh, 0.8)] * n_ns
    fc_fin_ns = [(r_ns, g_ns, b_ns, 0.8)] * n_ns
    ec_fin    = [(1.0,  1.0,  1.0,  0.4)] * n_ns
    ax.scatter(x_ns, bns_mfinal, marker=top_half,
               facecolors=fc_fin_bh, edgecolors=ec_fin, linewidths=0.5, s=s_ns_mfinal, zorder=4)
    ax.scatter(x_ns, bns_mfinal, marker=bot_half,
               facecolors=fc_fin_ns, edgecolors=ec_fin, linewidths=0.5, s=s_ns_mfinal, zorder=4)

    # ─── Axes ────────────────────────────────────────────────────────────────
    ax.set_yscale("log")
    ax.set_xlim(-1, arch_n)
    ax.set_ylim(0.9, 600)
    ax.set_xticks([])

    yticks = [1, 2, 5, 10, 20, 50, 100, 200]
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(y) for y in yticks], color="white", fontsize=25, alpha=0.5)
    ax.set_ylabel("Solar Masses", color="white", fontsize=25, alpha=0.5)

    ax.yaxis.grid(True, alpha=0.08, ls="-", lw=0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", which="both", length=0, colors="white")
    for t in yticks:
        ax.axhline(y=t, color="white", alpha=0.2)

    # ─── Title & legend ─────────────────────────────────────────────────────
    ax.set_title("Masses in the Stellar Graveyard",
                 color="white", fontsize=35, y=0.92)
    ax.text(0.55, 0.9, "Black Holes", transform=ax.transAxes,
            color=BH_COLOR, fontsize=14, ha="center", va="center")
    ax.text(0.4, 0.9, "One day of observations in 3G", fontsize=14, color="white",
            transform=ax.transAxes, ha="center", va="center")
    ax.text(0.66, 0.9, "  Neutron Stars", transform=ax.transAxes,
            color=NS_COLOR, fontsize=14, ha="center", va="center")
    ax.text(0.99, 4e-3, "Inspired by ``LIGO-Virgo-KAGRA masses in the stellar graveyard''",
            transform=ax.transAxes, color="white", fontsize=7, va="center", ha="right", alpha=0.6)
    ax.text(0.01, 4e-3, r"Harsh Narola \textbar{} Utrecht University \textbar{} Nikhef",
            color="white", alpha=1, va="center", ha="left",
            transform=ax.transAxes, fontsize=7)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig("et_masses_in_graveyard.pdf", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.savefig("et_masses_in_graveyard.png", dpi=500, bbox_inches="tight",
                facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
