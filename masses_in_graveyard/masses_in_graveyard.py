
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.path as mpath
from scipy.stats import gaussian_kde
import scienceplots
plt.style.use(['science'])

# ─── LVK black hole masses (GWTC-1/2/3 BBH events) ───────────────────────────
# Median component masses in solar masses — replace with your own catalog data
lvk_blackhole_mass1 = np.array([
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

num_bbh = int(300)
px_mass, x_mass = np.histogram(lvk_blackhole_mass1, density = 1)
third_gen_blackhole_mass1 = gaussian_kde(lvk_blackhole_mass1).resample(num_bbh, seed=2026)[0]

#np.random.choice(x_mass[:-1], p=px_mass, size=num_bbh)
q = np.random.uniform(1, 12, num_bbh)
third_gen_blackhole_mass2 = third_gen_blackhole_mass1 / q

_filter = np.where((third_gen_blackhole_mass1>5) & (third_gen_blackhole_mass2>5))
third_gen_blackhole_mass1 = third_gen_blackhole_mass1[_filter]
third_gen_blackhole_mass2 = third_gen_blackhole_mass2[_filter]

num_bns = 300
third_gen_neutron_star_mass1  = np.random.uniform(1.18, 2.25, num_bns)
third_gen_neutron_star_ratio  = np.random.uniform(0.8, 0.97, num_bns)
third_gen_neutron_star_mass2  = third_gen_neutron_star_mass1 * third_gen_neutron_star_ratio
_bns_filter = (third_gen_neutron_star_mass1 > 1) & (third_gen_neutron_star_mass2 > 1)
third_gen_neutron_star_mass1  = third_gen_neutron_star_mass1[_bns_filter]
third_gen_neutron_star_mass2  = third_gen_neutron_star_mass2[_bns_filter]
third_gen_neutron_star_mfinal = 0.95 * (third_gen_neutron_star_mass1 + third_gen_neutron_star_mass2)


num_nsbh = 300
third_gen_nsbh_ns     = np.random.uniform(1.18, 2.25, num_nsbh)
third_gen_nsbh_bh     = np.random.uniform(3, 30, num_nsbh)
third_gen_nsbh_mfinal = 0.95 * (third_gen_nsbh_ns + third_gen_nsbh_bh)

lvk_blackhole_mass2 = np.array([
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

lvk_mfinal = 0.95 * (lvk_blackhole_mass1 + lvk_blackhole_mass2)

# ─── Merge LVK + third-gen into one combined population ───────────────────────
n_lvk      = len(lvk_blackhole_mass1)
all_mass1  = np.concatenate([lvk_blackhole_mass1, third_gen_blackhole_mass1])
all_mass2  = np.concatenate([lvk_blackhole_mass2, third_gen_blackhole_mass2])
all_mfinal = 0.95 * (all_mass1 + all_mass2)
n_all      = len(all_mass1)

# ─── Colours ──────────────────────────────────────────────────────────────────
BH_COLOR        = "#008FBF"   # teal / cyan — GWTC black holes
NS_COLOR        = "#FF8C00"  #ornage
THIRDGEN_COLOR  = "#FF8C00"   # orange — 3rd-gen population
BG_COLOR        = "#0a0a0a"
MASS_SCALE = 5.0   # marker area = MASS_SCALE * mass  (area ∝ mass)

def ms(mass):
    """Marker size (area in pt²) proportional to mass."""
    return 2 * mass

def scatter_marker(ax, x, y, color, size, fill_alpha=0.7, edge_lw=0.8, **kwargs):
    """Scatter with a border: same color as fill but at full alpha."""
    r, g, b = mcolors.to_rgb(color)
    ax.scatter(x, y,
               facecolors=(r, g, b, fill_alpha),
               edgecolors=(r, g, b, 1.0),
               linewidths=edge_lw,
               s=size, **kwargs)

# ─── Arch ordering ────────────────────────────────────────────────────────────
# Top half (highest total mass) → arch: interleave so heaviest is in the centre.
# Bottom half (lowest total mass) → scattered randomly along x-axis.
total_mass = all_mass1 + all_mass2
sorted_idx = np.argsort(total_mass)          # ascending: lightest first

n_half    = n_all // 2
low_idx   = sorted_idx[:n_half]              # lightest half → scatter
high_idx  = sorted_idx[n_half:]              # heaviest half → arch

x_of = np.empty(n_all, dtype=float)

# Arch: lightest of the top-half go to edges, heaviest to centre
arch_n      = len(high_idx)
left, right = 0, arch_n - 1
for k, ev_idx in enumerate(high_idx):
    if k % 2 == 0:
        x_of[ev_idx] = left;  left  += 1
    else:
        x_of[ev_idx] = right; right -= 1

# Scatter: random x within the same x-range as the arch
rng = np.random.default_rng(42)
x_of[low_idx] = rng.uniform(0, arch_n - 1, size=n_half)

# ─── X positions for NSBH and BNS (placeholders) ──────────────────────────────
# Will be filled in when those plotting blocks are implemented.

# ─── Plot ─────────────────────────────────────────────────────────────────────
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(12, 10))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# ── All black holes (LVK + third-gen, unified style) ─────────────────────────
from matplotlib.collections import LineCollection

r, g, b = mcolors.to_rgb(BH_COLOR)

# Connecting lines m2 → m1 (vectorized)
segs = [[(x_of[i], all_mass2[i]), (x_of[i], all_mass1[i])] for i in range(n_all)]
ax.add_collection(LineCollection(segs, colors="white", linewidths=0.6, alpha=0.2, zorder=1))

# Arrows m1 → mfinal
for i in range(n_all):
    ax.annotate("", xy=(x_of[i], all_mfinal[i]), xytext=(x_of[i], all_mass1[i]),
                arrowprops=dict(arrowstyle="->", color="white", lw=0.5, alpha=0.5,
                                shrinkA=0, shrinkB=0),
                zorder=2)

# Mass markers (vectorized — 3 scatter calls total)
s_m2     = [ms(m) for m in all_mass2]
s_m1     = [ms(m) for m in all_mass1]
s_mfinal = [ms(m) for m in all_mfinal]
fc       = [(r, g, b, 0.5)] * n_all
ec       = [(r, g, b, 1.0)] * n_all
ax.scatter(x_of, all_mass2,   facecolors=fc,     edgecolors=ec, linewidths=0.8, s=s_m2,     zorder=3)
ax.scatter(x_of, all_mass1,   facecolors=fc,     edgecolors=ec, linewidths=0.8, s=s_m1,     zorder=3)
ax.scatter(x_of, all_mfinal,  facecolors=fc, edgecolors=ec, linewidths=0.8, s=s_mfinal, zorder=3)

# ── NSBH: third-gen NS-BH systems, scattered randomly along x ────────────────
n_nsbh    = len(third_gen_nsbh_ns)
rng_nsbh  = np.random.default_rng(77)
x_nsbh    = rng_nsbh.uniform(0, arch_n - 1, size=n_nsbh)

r_bh, g_bh, b_bh = mcolors.to_rgb(BH_COLOR)
r_ns2, g_ns2, b_ns2 = mcolors.to_rgb(NS_COLOR)

segs_nsbh = [[(x_nsbh[i], third_gen_nsbh_ns[i]), (x_nsbh[i], third_gen_nsbh_bh[i])]
             for i in range(n_nsbh)]
ax.add_collection(LineCollection(segs_nsbh, colors="white", linewidths=0.6, alpha=0.2, zorder=1))

for i in range(n_nsbh):
    ax.annotate("", xy=(x_nsbh[i], third_gen_nsbh_mfinal[i]),
                    xytext=(x_nsbh[i], third_gen_nsbh_bh[i]),
                arrowprops=dict(arrowstyle="->", color="white", lw=0.5, alpha=0.4,
                                shrinkA=0, shrinkB=0),
                zorder=2)

s_nsbh_ns     = [ms(m) for m in third_gen_nsbh_ns]
s_nsbh_bh     = [ms(m) for m in third_gen_nsbh_bh]
s_nsbh_mfinal = [ms(m) for m in third_gen_nsbh_mfinal]
fc_nsbh_ns  = [(r_ns2, g_ns2, b_ns2, 0.2)] * n_nsbh
ec_nsbh_ns  = [(r_ns2, g_ns2, b_ns2, 1.0)] * n_nsbh
fc_nsbh_bh  = [(r_bh,  g_bh,  b_bh,  0.2)] * n_nsbh
ec_nsbh_bh  = [(r_bh,  g_bh,  b_bh,  1.0)] * n_nsbh
ax.scatter(x_nsbh, third_gen_nsbh_ns,     facecolors=fc_nsbh_ns, edgecolors=ec_nsbh_ns,
           linewidths=0.8, s=s_nsbh_ns,     zorder=3)
ax.scatter(x_nsbh, third_gen_nsbh_bh,     facecolors=fc_nsbh_bh, edgecolors=ec_nsbh_bh,
           linewidths=0.8, s=s_nsbh_bh,     zorder=3)
ax.scatter(x_nsbh, third_gen_nsbh_mfinal, facecolors=fc_nsbh_bh, edgecolors=ec_nsbh_bh,
           linewidths=0.8, s=s_nsbh_mfinal, zorder=3)

# ── BNS: third-gen neutron stars, scattered randomly along x ─────────────────
n_ns   = len(third_gen_neutron_star_mass1)
rng_ns = np.random.default_rng(99)
x_ns   = rng_ns.uniform(0, arch_n - 1, size=n_ns)

# Place the 10 lowest-mass2 NS events near the centre
_ns_low10 = np.argsort(third_gen_neutron_star_mass2)[:10]
_center   = (arch_n - 1) / 2
x_ns[_ns_low10] = rng_ns.uniform(_center - 10, _center + 10, size=10)

r_ns, g_ns, b_ns = mcolors.to_rgb(NS_COLOR)

segs_ns = [[(x_ns[i], third_gen_neutron_star_mass2[i]), (x_ns[i], third_gen_neutron_star_mass1[i])]
           for i in range(n_ns)]
ax.add_collection(LineCollection(segs_ns, colors="white", linewidths=0.6, alpha=0.2, zorder=1))

s_ns_m2     = [ms(m) for m in third_gen_neutron_star_mass2]
s_ns_m1     = [ms(m) for m in third_gen_neutron_star_mass1]
s_ns_mfinal = [ms(m) for m in third_gen_neutron_star_mfinal]
fc_ns = [(r_ns, g_ns, b_ns, 0.2)] * n_ns
ec_ns = [(r_ns, g_ns, b_ns, 1.0)] * n_ns
ax.scatter(x_ns, third_gen_neutron_star_mass2, facecolors=fc_ns, edgecolors=ec_ns,
           linewidths=0.8, s=s_ns_m2, zorder=3)
ax.scatter(x_ns, third_gen_neutron_star_mass1, facecolors=fc_ns, edgecolors=ec_ns,
           linewidths=0.8, s=s_ns_m1, zorder=3)

# mfinal: split marker — top half BH_COLOR, bottom half NS_COLOR
_t = np.linspace(0, np.pi, 50)
_top_verts = np.vstack([[-1, 0], np.c_[np.cos(_t), np.sin(_t)], [1, 0]])
top_half = mpath.Path(_top_verts,
                      [mpath.Path.MOVETO] + [mpath.Path.LINETO]*50 + [mpath.Path.CLOSEPOLY])
_t = np.linspace(np.pi, 2*np.pi, 50)
_bot_verts = np.vstack([[1, 0], np.c_[np.cos(_t), np.sin(_t)], [-1, 0]])
bot_half = mpath.Path(_bot_verts,
                      [mpath.Path.MOVETO] + [mpath.Path.LINETO]*50 + [mpath.Path.CLOSEPOLY])

r_bh, g_bh, b_bh = mcolors.to_rgb(BH_COLOR)
fc_fin_bh = [(r_bh, g_bh, b_bh, 0.8)] * n_ns
fc_fin_ns = [(r_ns, g_ns, b_ns, 0.8)] * n_ns
ec_fin    = [(1.0,  1.0,  1.0,  0.4)] * n_ns
ax.scatter(x_ns, third_gen_neutron_star_mfinal, marker=top_half,
           facecolors=fc_fin_bh, edgecolors=ec_fin, linewidths=0.5, s=s_ns_mfinal, zorder=4)
ax.scatter(x_ns, third_gen_neutron_star_mfinal, marker=bot_half,
           facecolors=fc_fin_ns, edgecolors=ec_fin, linewidths=0.5, s=s_ns_mfinal, zorder=4)

for i in range(n_ns):
    ax.annotate("", xy=(x_ns[i], third_gen_neutron_star_mfinal[i]),
                    xytext=(x_ns[i], third_gen_neutron_star_mass1[i]),
                arrowprops=dict(arrowstyle="->", color="white", lw=0.5, alpha=0.4,
                                shrinkA=0, shrinkB=0),
                zorder=2)

# ─── Axes ─────────────────────────────────────────────────────────────────────
ax.set_yscale("log")
ax.set_xlim(-1, arch_n)
ax.set_ylim(0.9, 600)
ax.set_xticks([])

yticks = [1, 2, 5, 10, 20, 50, 100, 200]
ax.set_yticks(yticks)
ax.set_yticklabels([str(y) for y in yticks], color="white", fontsize=25, alpha = 0.5)
ax.set_ylabel("Solar Masses", color="white", fontsize=25, alpha = 0.5)

ax.yaxis.grid(True, alpha=0.08, ls="-", lw=0.5)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(axis="both", which="both", length=0, colors="white")

# ─── Title & legend ───────────────────────────────────────────────────────────
ax.set_title("Masses in the Stellar Graveyard",
             color="white", fontsize=35, y=0.92)
ax.text(0.55, 0.875, "Black Holes", transform=ax.transAxes,
        color=BH_COLOR, fontsize=14, ha="center", va="center")

ax.text(0.35, 0.875, "One day of observations in 3G", fontsize = 14, color = 'white', transform=ax.transAxes,
        ha="center", va="center")

ax.text(0.68, 0.875, "  Neutron Stars", transform=ax.transAxes,
        color=NS_COLOR, fontsize=14, ha="center", va="center")
for t in yticks:
    ax.axhline(y = t, color = 'white', alpha = 0.2)
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
#plt.show()
#plt.show()


#1e5 bbh per year snr>10
#1e5 bns per year snr>10
#1e5 nsbh per year (guess work)