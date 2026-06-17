import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.path as mpath
from matplotlib.collections import LineCollection
from scipy.stats import gaussian_kde
import scienceplots

plt.style.use(['science'])

# ─── LVK black hole masses (GWTC-1/2/3 BBH events) ───────────────────────────
# Median component masses in solar masses — replace with your own catalog data
LVK_BLACKHOLE_MASS1 = np.array([
    # GWTC-1
    35.6, 23.3, 13.7, 31.0, 10.9, 50.6, 35.2, 30.7, 35.5, 39.6,
    # GWTC-2
    24.6, 30.1, 34.8, 47.6, 40.0, 40.5, 46.2, 23.3, 36.0, 39.7,
    37.5, 66.0, 95.3, 41.2, 32.5, 69.1, 57.1, 38.4, 54.0, 68.1,
    11.6, 18.2, 36.4, 13.9, 37.7, 12.3, 41.6, 37.0, 32.1, 24.3,
    35.5, 44.0, 35.9,  8.9, 20.4, 35.2,
    # GWTC-3
    10.1, 22.6, 38.1, 40.1, 19.3, 13.1,
])

LVK_BLACKHOLE_MASS2 = np.array([
    # GWTC-1
    30.6, 13.6,  7.7, 20.1,  7.6, 34.3, 23.8, 25.3, 26.8, 29.4,
    # GWTC-2
    18.4,  8.3, 23.0, 28.4, 32.1, 32.2, 29.3, 12.6, 17.9, 26.5,
    25.3, 40.5, 69.0, 32.6, 21.0, 47.8, 35.8, 29.3, 40.8, 40.2,
     7.9, 12.0, 23.3,  8.2, 27.2,  8.1, 28.4, 27.0, 26.4, 11.2,
    19.5, 32.6, 25.0,  5.0, 15.5, 21.5,
    # GWTC-3
     7.3, 14.5, 20.8, 31.1, 13.9,  7.8,
])

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
    third_gen_mass1 = gaussian_kde(LVK_BLACKHOLE_MASS1).resample(num_bbh, seed=2026)[0]
    q = np.random.uniform(1, 12, num_bbh)
    third_gen_mass2 = third_gen_mass1 / q

    _filter = np.where((third_gen_mass1 > 5) & (third_gen_mass2 > 5))
    third_gen_mass1 = third_gen_mass1[_filter]
    third_gen_mass2 = third_gen_mass2[_filter]

    mass1 = np.concatenate([LVK_BLACKHOLE_MASS1, third_gen_mass1])
    mass2 = np.concatenate([LVK_BLACKHOLE_MASS2, third_gen_mass2])
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

    n_half   = n // 2
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
    fig, ax = plt.subplots(figsize=(12, 10))
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
    ax.text(0.55, 0.875, "Black Holes", transform=ax.transAxes,
            color=BH_COLOR, fontsize=14, ha="center", va="center")
    ax.text(0.35, 0.875, "One day of observations in 3G", fontsize=14, color="white",
            transform=ax.transAxes, ha="center", va="center")
    ax.text(0.68, 0.875, "  Neutron Stars", transform=ax.transAxes,
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
