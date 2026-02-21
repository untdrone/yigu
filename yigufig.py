import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# =========================
# Config (edit these)
# =========================
CSV_PATH = "spx_raw2025.csv"   # your 5-min history file
OUT_DIR = "day_pics"
TZ = "America/Chicago"

# Your data uses 08:30 CT start
TODAY_START = "08:30"
TODAY_END   = "15:00"          # SPX cash close = 15:00 CT (4pm ET)

# Yesterday's last hour (14:00–15:00 CT)
YDAY_LAST_HOUR_START = "14:00"
YDAY_LAST_HOUR_END   = "15:00"

# First 6 bars: 08:30, 08:35, 08:40, 08:45, 08:50, 08:55
FIRST6_START = "08:30"
FIRST6_END   = "08:55"

# =========================
# Time parsing (handles mixed tz)
# =========================
def find_time_column(df: pd.DataFrame) -> str:
    for c in ["time", "datetime", "timestamp", "date", "dt"]:
        if c in df.columns:
            return c
    raise ValueError(f"No timestamp column found. Columns: {df.columns.tolist()}")

def parse_mixed_timezone_to_chicago(s: pd.Series, tz: str = TZ) -> pd.Series:
    s = s.astype(str).str.strip()
    # non-capturing group avoids pandas warning about match groups
    tz_aware = s.str.contains(r"(?:Z|[+\-]\d{2}:\d{2})$", regex=True)

    aware = pd.to_datetime(s[tz_aware], errors="coerce", utc=True).dt.tz_convert(tz)
    naive = pd.to_datetime(s[~tz_aware], errors="coerce").dt.tz_localize(
        tz, ambiguous="infer", nonexistent="shift_forward"
    )

    out = pd.Series(index=s.index, dtype=f"datetime64[ns, {tz}]")
    out.loc[tz_aware] = aware
    out.loc[~tz_aware] = naive
    return out

# =========================
# Label logic  fff
# =========================
WEIGHTS = [1, 2, 4, 8, 16, 32]

def compute_first6_label(day_df: pd.DataFrame) -> tuple[int, str]:
    first6 = day_df[(day_df["clock"] >= FIRST6_START) & (day_df["clock"] <= FIRST6_END)].copy()
    first6 = first6.sort_values("time")

    if len(first6) < 6:
        raise ValueError("Not enough bars for first 6 candles (need 08:30..08:55).")

    first6 = first6.iloc[:6]
    bits = []

    for _, r in first6.iterrows():
        # bit = 0 if open > close
        # bit = 1 otherwise
        bits.append(0 if float(r["open"]) > float(r["close"]) else 1)

    label = int(sum(w * b for w, b in zip(WEIGHTS, bits)))
    bitstring = "".join(str(b) for b in bits)
    return label, bitstring

# =========================
# Candlestick plot (matplotlib, no extra libs)
# =========================
def plot_candles(df_plot: pd.DataFrame, split_index: int, title: str, outpath: str):
    """
    df_plot: rows in chronological order (yday last hour + today session)
    split_index: index position where today starts (vertical separator)
    """
    o = df_plot["open"].astype(float).to_numpy()
    h = df_plot["high"].astype(float).to_numpy()
    l = df_plot["low"].astype(float).to_numpy()
    c = df_plot["close"].astype(float).to_numpy()
    labels = df_plot["time"].dt.strftime("%m-%d %H:%M").tolist()

    x = np.arange(len(df_plot))
    fig, ax = plt.subplots(figsize=(16, 6), dpi=140)

    # Wicks
    ax.vlines(x, l, h, linewidth=1)

    # Bodies
    # Bodies + wicks (colored consistently)
    width = 0.7
    for i in range(len(df_plot)):
        # Standard candle rule:
        #   green if open < close
        #   red   if open > close
        color = "green" if o[i] < c[i] else "red"

        # Wick (high-low)
        ax.vlines(x[i], l[i], h[i], linewidth=1.2, color=color)

        # Body
        y0 = min(o[i], c[i])
        height = abs(c[i] - o[i])
        if height == 0:
            height = 0.0001  # tiny body for doji

        rect = Rectangle(
            (x[i] - width / 2, y0),
            width,
            height,
            facecolor=color,
            edgecolor=color
        )
        ax.add_patch(rect)


    # Separator between yesterday and today
    ax.axvline(split_index - 0.5, linestyle="--", linewidth=1)

    ax.set_title(title)
    ax.set_xlim(-1, len(df_plot))

    # Fewer x ticks
    step = max(1, len(df_plot) // 12)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([labels[i] for i in range(0, len(labels), step)], rotation=45, ha="right")

    ax.grid(True, linewidth=0.5, alpha=0.4)
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

# =========================
# Main pipeline
# =========================
def main(target_label: int = 33):
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.lower().str.strip()

    tcol = find_time_column(df)
    df["time"] = parse_mixed_timezone_to_chicago(df[tcol], TZ)

    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    df = df.dropna(subset=["time"]).sort_values("time")
    df["date"] = df["time"].dt.date
    df["clock"] = df["time"].dt.strftime("%H:%M")

    days = sorted(df["date"].unique())
    if len(days) < 2:
        raise ValueError("Need at least 2 days of data (for yesterday last hour + today).")

    matched = []

    # Start at day 2 because we need yesterday
    for i in range(1, len(days)):
        d_today = days[i]
        d_yday = days[i - 1]

        today = df[df["date"] == d_today].copy()
        yday = df[df["date"] == d_yday].copy()

        # Compute label from today's first 6 bars starting 08:30
        try:
            label, bits = compute_first6_label(today)
        except Exception:
            continue

        if label != target_label:
            continue

        # Build plot window: yesterday last hour + today session
        y_last = yday[(yday["clock"] >= YDAY_LAST_HOUR_START) & (yday["clock"] <= YDAY_LAST_HOUR_END)].copy()
        t_sess = today[(today["clock"] >= TODAY_START) & (today["clock"] <= TODAY_END)].copy()

        if y_last.empty or t_sess.empty:
            continue

        y_last = y_last.sort_values("time")
        t_sess = t_sess.sort_values("time")

        df_plot = pd.concat([y_last, t_sess], ignore_index=True)
        split_index = len(y_last)

        title = f"SPX 5m | {d_today} | label={label} bits={bits} (b1..b6)"
        outpath = os.path.join(OUT_DIR, f"{d_today}_label{label}_bits{bits}.png")

        plot_candles(df_plot, split_index, title, outpath)
        matched.append((str(d_today), label, bits, outpath))

    print(f"Target label = {target_label}")
    print("Matched days:")
    if not matched:
        print("  (none found)")
    else:
        for d, lab, bits, path in matched:
            print(f"  {d}  label={lab}  bits={bits}  -> {path}")

if __name__ == "__main__":
    # Improvement: allow command-line label, e.g.:
    #   python plot_days_by_label.py 33
    lab = int(sys.argv[1]) if len(sys.argv) > 1 else 33
    main(target_label=lab)
