"""Plot a lever session recorded by the Bonsai workflow.

Inputs are the two CSVs the workflow writes (same start-time suffix):
    encoder_trace<suffix>.csv   Timestamp,Value   ~1 kHz encoder samples
    events<suffix>.csv          Timestamp,Value   beep | home | return

Usage:
    python plot_session.py                      # newest pair in ~/Desktop
    python plot_session.py DIR_OR_SUFFIX_PATH   # newest pair in a directory
    python plot_session.py --encoder A.csv --events B.csv --out figs/
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TARGET = 60          # encoder position the workflow returns to (60 - encoder)
THRESHOLD = 200      # encoder value that triggers the beep
STEP_TOL = 2         # |error| below this -> no return move
COLORS = {"beep": "tab:red", "home": "tab:green", "return": "tab:blue"}


def find_pair(directory):
    """Newest encoder_trace*/events* pair sharing a suffix."""
    traces = sorted(Path(directory).expanduser().glob("encoder_trace*.csv"))
    for trace in reversed(traces):
        events = trace.with_name(trace.name.replace("encoder_trace", "events", 1))
        if events.exists():
            return trace, events
    raise FileNotFoundError(f"no encoder_trace/events pair in {directory}")


def load(encoder_path, events_path):
    enc = pd.read_csv(encoder_path)
    ev = pd.read_csv(events_path)
    for df in (enc, ev):
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="ISO8601", utc=True)
    t0 = enc["Timestamp"].iloc[0]
    enc["t"] = (enc["Timestamp"] - t0).dt.total_seconds()
    ev["t"] = (ev["Timestamp"] - t0).dt.total_seconds()
    return enc[["t", "Value"]], ev[["t", "Value"]].rename(columns={"Value": "event"})


def add_trials(ev):
    """Trial n = everything from beep n up to the next beep (trial 0 = before the first)."""
    ev = ev.copy()
    ev["trial"] = (ev["event"] == "beep").cumsum()
    return ev


def mark_events(ax, ev, labels=True, alpha=0.6):
    for name, color in COLORS.items():
        times = ev.loc[ev["event"] == name, "t"]
        ax.vlines(times, 0, 1, transform=ax.get_xaxis_transform(), color=color,
                  lw=1.2, alpha=alpha, label=name if labels else None)


def plot_overview(enc, ev, out):
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(enc["t"], enc["Value"], lw=0.7, color="k")
    ax.axhline(THRESHOLD, color=COLORS["beep"], ls="--", lw=0.8, label=f"beep threshold ({THRESHOLD})")
    ax.axhline(TARGET, color=COLORS["return"], ls="--", lw=0.8, label=f"return target ({TARGET})")
    mark_events(ax, ev)
    ax.set(xlabel="time (s)", ylabel="encoder value", title="Encoder trace with events")
    ax.legend(loc="upper right", ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "1_overview.png", dpi=150)
    plt.close(fig)


def plot_trial_grid(enc, ev, out, cols=3):
    """One panel per beep-to-beep trial, sharing the y axis."""
    beeps = ev.loc[ev["event"] == "beep", "t"].to_numpy()
    if len(beeps) == 0:
        return
    ends = np.append(beeps[1:], enc["t"].iloc[-1])
    rows = int(np.ceil(len(beeps) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 2.6 * rows), sharey=True, squeeze=False)
    for ax, (i, (a, b)) in zip(axes.flat, enumerate(zip(beeps, ends))):
        seg = enc[(enc["t"] >= a - 0.5) & (enc["t"] <= b)]
        ax.plot(seg["t"] - a, seg["Value"], lw=0.8, color="k")
        sub = ev[(ev["t"] >= a - 0.5) & (ev["t"] <= b)].assign(t=lambda d: d["t"] - a)
        mark_events(ax, sub, labels=False)
        ax.axhline(TARGET, color=COLORS["return"], ls=":", lw=0.7)
        ax.set_title(f"trial {i + 1}  (beep at {a:.1f} s)", fontsize=9)
    for ax in axes.flat[len(beeps):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("time from beep (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("encoder")
    handles = [plt.Line2D([], [], color=c, label=n) for n, c in COLORS.items()]
    fig.legend(handles=handles, loc="upper right", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "2_trials.png", dpi=150)
    plt.close(fig)


def plot_beep_aligned(enc, ev, out, pre=1.0, post=2.5):
    """All trials overlaid, time-locked to the beep (the >200 crossing)."""
    beeps = ev.loc[ev["event"] == "beep", "t"].to_numpy()
    if len(beeps) == 0:
        return
    cmap = plt.cm.viridis(np.linspace(0, 1, len(beeps)))
    fig, ax = plt.subplots(figsize=(10, 5))
    for color, b in zip(cmap, beeps):
        seg = enc[(enc["t"] >= b - pre) & (enc["t"] <= b + post)]
        ax.plot(seg["t"] - b, seg["Value"], lw=0.8, color=color, alpha=0.8)
    ax.axvline(0, color=COLORS["beep"], lw=1)
    ax.axvspan(0, 0.5, color=COLORS["beep"], alpha=0.1, label="tone (0.5 s)")
    ax.axhline(THRESHOLD, color=COLORS["beep"], ls="--", lw=0.8)
    ax.axhline(TARGET, color=COLORS["return"], ls="--", lw=0.8)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(1, len(beeps)))
    fig.colorbar(sm, ax=ax, label="trial")
    ax.set(xlabel="time from beep (s)", ylabel="encoder value", title="Trials aligned to beep")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "3_beep_aligned.png", dpi=150)
    plt.close(fig)


def trial_metrics(enc, ev):
    """Per-trial numbers; the quantities the plots below are drawn from."""
    ev = add_trials(ev)
    t, v = enc["t"].to_numpy(), enc["Value"].to_numpy()

    def at(x):
        return v[min(np.searchsorted(t, x), len(v) - 1)]

    rows = []
    for n, g in ev[ev["trial"] > 0].groupby("trial"):
        beep = g.loc[g["event"] == "beep", "t"].iloc[0]
        rets = g.loc[g["event"] == "return", "t"]
        homes = g.loc[g["event"] == "home", "t"]
        rows.append({
            "trial": n,
            "beep_t": beep,
            "pre_beep_pos": at(beep - 0.1),
            "peak": v[(t >= beep) & (t <= beep + 0.5)].max(),
            "n_home": len(homes),
            "n_return": len(rets),
            "first_home_latency": homes.iloc[0] - beep if len(homes) else np.nan,
            "final_pos_before_next_beep": at(g["t"].max() + 0.5),
        })
    return pd.DataFrame(rows).set_index("trial")


def return_table(enc, ev, settle=(0.25, 0.45)):
    """For each home->return pair: encoder after the home reset and after the return move."""
    t, v = enc["t"].to_numpy(), enc["Value"].to_numpy()
    ev = ev.reset_index(drop=True)
    rows = []
    for i, r in ev[ev["event"] == "return"].iterrows():
        homes = ev.loc[:i - 1].query("event == 'home'")
        if homes.empty:
            continue
        h = homes["t"].iloc[-1]
        nxt = ev["t"].iloc[i + 1] if i + 1 < len(ev) else t[-1]
        a, b = r["t"] + settle[0], min(r["t"] + settle[1], nxt)
        win = v[(t >= a) & (t <= b)]
        rows.append({
            "home_t": h, "return_t": r["t"], "latency_ms": (r["t"] - h) * 1000,
            "enc_after_home": v[min(np.searchsorted(t, h + 0.1), len(v) - 1)],
            "settled": np.median(win) if len(win) else np.nan,
        })
    return pd.DataFrame(rows)


def plot_summary(enc, ev, out):
    m = trial_metrics(enc, ev)
    rt = return_table(enc, ev)
    beeps = ev.loc[ev["event"] == "beep", "t"].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    ax = axes[0, 0]
    ax.plot(m.index, m["pre_beep_pos"], "o-", color="k")
    ax.set(xlabel="trial", ylabel="encoder 100 ms before beep", title="Approach position drift")

    ax = axes[0, 1]
    ax.plot(rt.index, rt["settled"], "o", color=COLORS["return"], label="after return")
    ax.plot(rt.index, rt["enc_after_home"], "x", color=COLORS["home"], label="100 ms after home")
    ax.axhline(TARGET, color="k", ls="--", lw=0.8)
    ax.axhspan(TARGET - STEP_TOL, TARGET + STEP_TOL, color="k", alpha=0.08)
    ax.set(xlabel="home→return pair #", ylabel="encoder", title=f"Pump positioning (target {TARGET})")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.hist(rt["latency_ms"], bins=np.arange(0, 200, 5), color=COLORS["return"])
    ax.set(xlabel="home→return latency (ms)",
           title="Mostly workflow delays (100 + 50 ms)")

    ax = axes[1, 1]
    ax.plot(np.arange(2, len(beeps) + 1), np.diff(beeps), "o-", color=COLORS["beep"])
    ax.set(xlabel="trial", ylabel="inter-beep interval (s)", title="Trial pacing")

    fig.tight_layout()
    fig.savefig(out / "4_summary.png", dpi=150)
    plt.close(fig)
    m.to_csv(out / "trial_metrics.csv")
    rt.to_csv(out / "return_table.csv", index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("directory", nargs="?", default="~/Desktop")
    p.add_argument("--encoder")
    p.add_argument("--events")
    p.add_argument("--out", default=None, help="output folder (default: figures/<session suffix>)")
    args = p.parse_args()

    if args.encoder and args.events:
        enc_path, ev_path = Path(args.encoder), Path(args.events)
    else:
        enc_path, ev_path = find_pair(args.directory)
    out = Path(args.out) if args.out else Path("figures") / enc_path.stem.replace("encoder_trace", "")
    out.mkdir(parents=True, exist_ok=True)

    enc, ev = load(enc_path, ev_path)
    print(f"{enc_path.name}: {len(enc)} samples, {enc['t'].iloc[-1]:.1f} s; "
          f"{ev_path.name}: {ev['event'].value_counts().to_dict()}")
    plot_overview(enc, ev, out)
    plot_trial_grid(enc, ev, out)
    plot_beep_aligned(enc, ev, out)
    plot_summary(enc, ev, out)
    print(f"wrote figures and tables to {out}/")


if __name__ == "__main__":
    main()
