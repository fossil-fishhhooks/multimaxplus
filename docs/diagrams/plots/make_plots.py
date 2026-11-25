"""Generate benchmark plots for multimaxplus vs upstream mumax+.
Usage: python3 -I make_plots.py <extracted_data_dir> <out_dir>"""
import json, csv, sys, re, statistics as st
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
for f in Path("/usr/share/fonts/opentype/inter").glob("Inter-*.otf"): fm.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "svg.fonttype": "none"})

INK, INK2, GRID, BASE = "#191919", "#666666", "#e7e7e7", "#c2c2c2"
C_ORIG, C_1GPU, C_2GPU, C_OFF = "#b4bac2", "#84b9f3", "#1b64be", "#f47230"
STAGE_COL = {"pack": "#257de0", "fwd": "#f47230", "mul": "#00b894", "inv": "#f2a600",
             "exch": "#ee7fac", "llg": "#008a17", "rk": "#6a4ce2"}

LATE = DATA / "benchmarks/local-rk-perf-20261125T025942Z"
EARLY = DATA / "multimaxplus/benchmarks/results/20261124T222332Z"
GRIDS = ["128x128x8", "1024x1024x1", "512x512x32", "768x768x32", "1024x1024x32"]
def cells(g): a, b, c = map(int, g.split("x")); return a*b*c
GRIDS.sort(key=cells)
def pretty(g):
    a, b, c = g.split("x"); return f"{a}²×{c}" if a == b else f"{a}×{b}×{c}"
def cellstr(g):
    n = cells(g); return f"{n/1e6:.1f} M cells" if n >= 1e6 else f"{n/1e3:.0f} k cells"

def js(p): return json.load(open(p))
rows = {}
for g in GRIDS:
    o = js(EARLY / f"timing/dynamics/{g}/original/summary.json")
    r = {"orig": o["median_ms"], "orig_b": o["batch_ms"]}
    for rk in (0, 1):
        n = js(LATE / f"rk-{rk}/{g}/normal/summary.json")
        m = js(LATE / f"rk-{rk}/{g}/cufftmp2/summary.json")
        r[f"one_rk{rk}"], r[f"one_rk{rk}_b"] = n["ms"], n["batch_ms"]
        r[f"two_rk{rk}"], r[f"two_rk{rk}_b"] = m["ms"], m["batch_ms"]
    # cross-check medians
    for k in ("orig_b", "one_rk1_b", "two_rk1_b", "two_rk0_b"):
        assert len(r[k]) == 5
    assert abs(st.median(r["two_rk1_b"]) - r["two_rk1"]) < 1e-6
    rows[g] = r

def err(b, med): return [[med - min(b)], [max(b) - med]]
def style(ax):
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASE); ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
def header(fig, title, sub):
    fig.text(0.035, 0.955, title, fontsize=19, fontweight="semibold", color=INK, va="top")
    fig.text(0.035, 0.895, sub, fontsize=11.5, color=INK2, va="top")
def footer(fig, txt): fig.text(0.035, 0.025, txt, fontsize=8.5, color=INK2, va="bottom", linespacing=1.5)
def legend(fig, items, x=0.035, y=0.835):
    for lab, col in items:
        fig.patches.append(plt.Rectangle((x, y-0.008), 0.012, 0.022, transform=fig.transFigure, color=col, lw=0))
        t = fig.text(x+0.017, y+0.003, lab, fontsize=11, color=INK, va="center")
        fig.canvas.draw(); w = t.get_window_extent().width / fig.bbox.width
        x += 0.017 + w + 0.025
def save(fig, name):
    fig.savefig(OUT/f"{name}.png", dpi=200, facecolor="white"); fig.savefig(OUT/f"{name}.svg", facecolor="white"); plt.close(fig)
HW = "2× NVIDIA A100 SXM4 80 GB (NVLink)"

# ---------- 1. timestep time ----------
fig, axs = plt.subplots(1, 5, figsize=(14, 5.6))
fig.subplots_adjust(left=0.03, right=0.985, top=0.71, bottom=0.11, wspace=0.18)
header(fig, "Time per RK timestep, by grid size", "Milliseconds per complete Fehlberg step (lower is better). Each panel has its own scale.")
legend(fig, [("Original mumax⁺, 1 GPU", C_ORIG), ("multimax⁺, 1 GPU", C_1GPU), ("multimax⁺, 2 GPUs (pruned demag FFT, local RK)", C_2GPU)])
for ax, g in zip(axs, GRIDS):
    r = rows[g]; vals = [r["orig"], r["one_rk1"], r["two_rk1"]]
    bs = [r["orig_b"], r["one_rk1_b"], r["two_rk1_b"]]
    cols = [C_ORIG, C_1GPU, C_2GPU]
    for i, (v, b, c) in enumerate(zip(vals, bs, cols)):
        ax.bar(i, v, width=0.78, color=c, zorder=2)
        ax.errorbar(i, v, yerr=err(b, v), color=INK, lw=0.8, capsize=2, zorder=3)
        ax.text(i, max(b)+max(vals)*0.025, f"{v:.1f}" if v >= 10 else f"{v:.2f}", ha="center", va="bottom", fontsize=10.5, color=INK, fontweight="semibold")
    ax.set_xlim(-0.6, 2.6); ax.set_ylim(0, max(vals)*1.14); style(ax)
    ax.set_title(pretty(g), fontsize=13.5, fontweight="semibold", color=INK, pad=22)
    ax.text(0.5, 1.02, cellstr(g), transform=ax.transAxes, ha="center", va="bottom", fontsize=9.5, color=INK2)
    ax.set_title(pretty(g), fontsize=13.5, fontweight="semibold", color=INK, pad=20)
footer(fig, HW)
save(fig, "01_timestep_time")

# ---------- 2. speedups ----------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.45, 1]})
fig.subplots_adjust(left=0.035, right=0.985, top=0.68, bottom=0.13, wspace=0.12)
header(fig, "Speedup", "")
x = list(range(len(GRIDS)))
sp_o1 = [rows[g]["orig"]/rows[g]["one_rk1"] for g in GRIDS]
sp_o2 = [rows[g]["orig"]/rows[g]["two_rk1"] for g in GRIDS]
sp_12 = [rows[g]["one_rk1"]/rows[g]["two_rk1"] for g in GRIDS]
w = 0.38
for i in x:
    for off, v, c in ((-w/2, sp_o1[i], C_1GPU), (w/2, sp_o2[i], C_2GPU)):
        a1.bar(i+off, v, width=w-0.02, color=c, zorder=2)
        a1.text(i+off, v+0.04, f"{v:.2f}×", ha="center", va="bottom", fontsize=10, color=INK, fontweight="semibold")
    a2.bar(i, sp_12[i], width=0.55, color=C_2GPU, zorder=2)
    a2.text(i, sp_12[i]+0.04, f"{sp_12[i]:.2f}×", ha="center", va="bottom", fontsize=10.5, color=INK, fontweight="semibold")
for ax, ttl, sub in ((a1, "vs. original upstream mumax⁺ (1 GPU)", "baseline = 1.0×"), (a2, "vs. multimax⁺ on 1 GPU", "2 GPUs ÷ 1 GPU")):
    ax.axhline(1, color=INK, lw=1, zorder=3); ax.set_ylim(0, 2.8); ax.set_xlim(-0.6, len(GRIDS)-0.4)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_yticks([]); ax.set_xticks(x)
    ax.set_xticklabels([f"{pretty(g)}" for g in GRIDS], fontsize=10.5, color=INK); ax.tick_params(bottom=False)
    ax.set_title(ttl, loc="left", fontsize=13, fontweight="semibold", color=INK, pad=22)
    ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=9.5, color=INK2, va="bottom")
a1.text(len(GRIDS)-0.42, 1.03, "1.0×", fontsize=9, color=INK2, ha="right", va="bottom")
legend(fig, [("multimax⁺, 1 GPU", C_1GPU), ("multimax⁺, 2 GPUs (pruned demag FFT, local RK)", C_2GPU)], y=0.865)
footer(fig, HW)
save(fig, "02_speedup")

# ---------- 3. local RK ----------
fig, axs = plt.subplots(1, 5, figsize=(14, 5.4))
fig.subplots_adjust(left=0.03, right=0.985, top=0.71, bottom=0.11, wspace=0.18)
header(fig, "Incremental benefit of local RK", "Two GPUs with pruned demag FFT; local RK off vs on. Milliseconds per RK step (lower is better).")
legend(fig, [("Local RK off", C_OFF), ("Local RK on", C_2GPU)])
for ax, g in zip(axs, GRIDS):
    r = rows[g]; v0, v1 = r["two_rk0"], r["two_rk1"]
    for i, (v, b, c) in enumerate(((v0, r["two_rk0_b"], C_OFF), (v1, r["two_rk1_b"], C_2GPU))):
        ax.bar(i, v, width=0.7, color=c, zorder=2)
        ax.errorbar(i, v, yerr=err(b, v), color=INK, lw=0.8, capsize=2, zorder=3)
        ax.text(i, max(b)+max(v0, v1)*0.025, f"{v:.1f}" if v >= 10 else f"{v:.2f}", ha="center", va="bottom", fontsize=10.5, color=INK, fontweight="semibold")
    gain = (v0-v1)/v0*100
    ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, max(v0, v1)*1.14); style(ax)
    ax.set_title(pretty(g), fontsize=13.5, fontweight="semibold", color=INK, pad=20)
    ax.text(0.5, 1.02, f"{gain:.0f}% faster · {v0/v1:.2f}×", transform=ax.transAxes, ha="center", va="bottom", fontsize=9.5, color=INK2)
footer(fig, HW)
save(fig, "03_local_rk")

# ---------- 4. breakdown ----------
P = LATE / "profile/512x512x32"
def parse(log, ranks=None):
    ent = []
    for ln in open(log):
        if ln.startswith("step-profile"):
            d = dict(re.findall(r"(\S+?)=([-\d.e]+)", ln)); ent.append({k: float(v) for k, v in d.items() if k not in ("rank",)} | {"rank": int(d["rank"])})
    return ent
def stages(log):
    e = [x for x in parse(log) if x["t"] > 2.5e-13]   # drop the 2 warmup steps (t=1e-13, 2e-13)
    assert len(e) >= 2
    m = lambda k: st.mean(x[k] for x in e)
    out = {"pack": m("demag-pad")+m("demag-crop")+m("demag-gather"), "fwd": m("demag-forward"), "mul": m("demag-multiply"), "inv": m("demag-inverse"),
           "exch": m("exchange"), "llg": m("llg"), "rk": m("stage-normalize-assign")+m("final-normalize-assign")}
    tot = m("rhs-inclusive")+m("stage-normalize-assign")+m("final-normalize-assign")+m("stage-build")+m("final-build")
    out["other"] = tot - sum(out.values()); out["total"] = tot
    return out
b1, b2 = stages(P/"normal/run.log"), stages(P/"cufftmp2/run.log")
labels = {"pack": "Demag pad/crop/gather", "fwd": "Demag forward*", "mul": "Demag multiply", "inv": "Demag inverse*",
          "exch": "Exchange", "llg": "LLG torque", "rk": "RK assignment", "other": "Other"}
STAGE_COL["other"] = "#c2c2c2"
order = ["pack", "fwd", "mul", "inv", "exch", "llg", "rk", "other"]
fig, ax = plt.subplots(figsize=(14, 5.4)); fig.subplots_adjust(left=0.14, right=0.97, top=0.68, bottom=0.19)
header(fig, "Where a timestep goes at 512²×32", "")
legend(fig, [(labels[k], STAGE_COL[k]) for k in order], y=0.865)
for yi, (name, b) in enumerate((("multimax⁺, 2 GPUs\n(per-GPU timeline)", b2), ("multimax⁺, 1 GPU", b1))):
    left = 0
    for k in order:
        v = b[k]; ax.barh(yi, v, left=left, height=0.58, color=STAGE_COL[k], edgecolor="white", linewidth=1.5, zorder=2)
        if v > 3.0 and k != "other": ax.text(left+v/2, yi, f"{v:.1f}", ha="center", va="center", fontsize=10, color="#0b0b0b" if k == "inv" else "white", fontweight="semibold")
        left += v
    ax.text(left+0.8, yi, f"{b['total']:.1f} ms", va="center", fontsize=11, color=INK, fontweight="semibold")
ax.set_yticks([0, 1]); ax.set_yticklabels(["multimax⁺, 2 GPUs\n(local RK on)", "multimax⁺, 1 GPU"], fontsize=11.5, color=INK)
ax.invert_yaxis(); ax.set_xlim(0, 75); ax.set_xlabel("ms per step (profiled)", fontsize=10, color=INK2, labelpad=8)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(BASE); ax.tick_params(axis="y", left=False); ax.tick_params(axis="x", colors=INK2, color=BASE)
ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
footer(fig, "Profiled run, so proportions only. * Includes packing and communication.")
save(fig, "04_breakdown_512x512x32")

# ---------- data table ----------
with open(OUT/"plot_data.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["grid", "original_1gpu_ms", "multimaxplus_1gpu_ms", "multimaxplus_2gpu_rk1_ms", "multimaxplus_2gpu_rk0_ms", "speedup_vs_original_multimaxplus_1gpu", "speedup_vs_original_multimaxplus_2gpu", "speedup_2gpu_vs_multimaxplus_1gpu", "localrk_gain_pct"])
    for g in GRIDS:
        r = rows[g]; w.writerow([g, r["orig"], r["one_rk1"], r["two_rk1"], r["two_rk0"], r["orig"]/r["one_rk1"], r["orig"]/r["two_rk1"], r["one_rk1"]/r["two_rk1"], (r["two_rk0"]-r["two_rk1"])/r["two_rk0"]*100])
print(open(OUT/"plot_data.csv").read()); print({k: round(v, 2) for k, v in b1.items()}); print({k: round(v, 2) for k, v in b2.items()})
