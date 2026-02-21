import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

# --- CONFIG ---
CSV_PATH = "spx_raw2025.csv"
TZ = "America/Chicago"
TODAY_START, TODAY_END = "08:30", "15:00"
YDAY_LAST_HOUR_START, YDAY_LAST_HOUR_END = "14:00", "15:00"
FIRST6_START, FIRST6_END = "08:30", "08:55"
WEIGHTS = [1, 2, 4, 8, 16, 32]

st.set_page_config(page_title="SPX Label Explorer", layout="wide")

# --- DATA PROCESSING (Cached for speed) ---
@st.cache_data
def load_and_preprocess():
    if not os.path.exists(CSV_PATH):
        st.error(f"File {CSV_PATH} not found!")
        return None
    
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.lower().str.strip()
    
    # Identify time column
    tcol = next((c for c in ["time", "datetime", "timestamp", "date", "dt"] if c in df.columns), None)
    
    # Simple conversion (Refactored from your script)
    df['time'] = pd.to_datetime(df[tcol]).dt.tz_localize(None).dt.tz_localize(TZ, ambiguous='infer')
    df = df.dropna(subset=["time"]).sort_values("time")
    df["date"] = df["time"].dt.date
    df["clock"] = df["time"].dt.strftime("%H:%M")
    return df

def compute_label(day_df):
    first6 = day_df[(day_df["clock"] >= FIRST6_START) & (day_df["clock"] <= FIRST6_END)].sort_values("time").iloc[:6]
    if len(first6) < 6: return None, None
    bits = [0 if float(r["open"]) > float(r["close"]) else 1 for _, r in first6.iterrows()]
    label = int(sum(w * b for w, b in zip(WEIGHTS, bits)))
    return label, "".join(str(b) for b in bits)

def plot_candles_st(df_plot, split_index, title):
    o, h, l, c = df_plot["open"].astype(float), df_plot["high"].astype(float), df_plot["low"].astype(float), df_plot["close"].astype(float)
    x = np.arange(len(df_plot))
    
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.7
    
    for i in range(len(df_plot)):
        color = "green" if o.iloc[i] < c.iloc[i] else "red"
        ax.vlines(x[i], l.iloc[i], h.iloc[i], color=color, linewidth=1)
        y0, height = min(o.iloc[i], c.iloc[i]), abs(c.iloc[i] - o.iloc[i])
        ax.add_patch(Rectangle((x[i] - width/2, y0), width, max(height, 0.001), facecolor=color))

    ax.axvline(split_index - 0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_title(title)
    
    # Optimization: Show fewer x-axis labels
    step = max(1, len(df_plot) // 10)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(df_plot["time"].dt.strftime("%m-%d %H:%M").iloc[::step], rotation=30)
    plt.grid(alpha=0.2)
    return fig

# --- WEB UI ---
st.title("📈 SPX Intraday Label Explorer")

df = load_and_preprocess()

if df is not None:
    target_label = st.number_input("Enter Integer Label (e.g. 33):", min_value=0, max_value=63, value=33)
    
    days = sorted(df["date"].unique())
    matches = []

    # Filter days matching label
    for i in range(1, len(days)):
        d_today, d_yday = days[i], days[i-1]
        today_data = df[df["date"] == d_today]
        label, bits = compute_label(today_data)
        
        if label == target_label:
            y_last = df[(df["date"] == d_yday) & (df["clock"] >= YDAY_LAST_HOUR_START) & (df["clock"] <= YDAY_LAST_HOUR_END)]
            t_sess = today_data[(today_data["clock"] >= TODAY_START) & (today_data["clock"] <= TODAY_END)]
            
            if not y_last.empty and not t_sess.empty:
                matches.append({
                    "date": d_today,
                    "bits": bits,
                    "plot_data": pd.concat([y_last, t_sess]),
                    "split": len(y_last)
                })

    st.subheader(f"Found {len(matches)} days with Label {target_label}")

    # Display plots
    for m in matches:
        with st.container():
            st.write(f"### Date: {m['date']} (Bits: {m['bits']})")
            fig = plot_candles_st(m['plot_data'], m['split'], f"SPX | {m['date']} | Label {target_label}")
            st.pyplot(fig)
            st.divider()