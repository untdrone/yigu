import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import base64

# --- CONFIG ---
CSV_PATH = "spx_raw2025.csv"
TZ = "America/Chicago"
TODAY_START, TODAY_END = "08:30", "15:00"
YDAY_LAST_HOUR_START, YDAY_LAST_HOUR_END = "14:00", "15:00"
FIRST6_START, FIRST6_END = "08:30", "08:55"
WEIGHTS = [1, 2, 4, 8, 16, 32]

# --- PAGE CONFIG ---
st.set_page_config(page_title="SPX Trend", layout="wide")

# --- DATA PROCESSING ---
@st.cache_data
def load_and_preprocess():
    if not os.path.exists(CSV_PATH):
        st.error(f"File {CSV_PATH} not found!")
        return None
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.lower().str.strip()
    tcol = next((c for c in ["time", "datetime", "timestamp", "date", "dt"] if c in df.columns), None)
    
    # Convert time
    df['time'] = pd.to_datetime(df[tcol]).dt.tz_localize(None).dt.tz_localize(TZ, ambiguous='infer')
    df = df.dropna(subset=["time"]).sort_values("time")
    
    # Calculate SMAs for the full dataset
    df['sma5'] = df['close'].rolling(window=5).mean()
    df['sma10'] = df['close'].rolling(window=10).mean()
    df['sma20'] = df['close'].rolling(window=20).mean()
    df['sma50'] = df['close'].rolling(window=50).mean()
    
    df["date"] = df["time"].dt.date
    df["clock"] = df["time"].dt.strftime("%H:%M")
    return df

def compute_label_from_df(day_df):
    first6 = day_df[(day_df["clock"] >= FIRST6_START) & (day_df["clock"] <= FIRST6_END)].sort_values("time").iloc[:6]
    if len(first6) < 6: return None, None
    bits = [0 if float(r["open"]) > float(r["close"]) else 1 for _, r in first6.iterrows()]
    label = int(sum(w * b for w, b in zip(WEIGHTS, bits)))
    return label, "".join(str(b) for b in bits)

def plot_candles_st(df_plot, split_index, title):
    o = df_plot["open"].astype(float)
    h = df_plot["high"].astype(float)
    l = df_plot["low"].astype(float)
    c = df_plot["close"].astype(float)
    
    x = np.arange(len(df_plot))
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # --- ADD SHADOW FOR YESTERDAY ---
    # Highlights the background from the start to the split line
    ax.axvspan(-0.5, split_index - 0.5, facecolor='gray', alpha=0.1, label='Yesterday (Last Hour)')

    # Plot SMAs
    ax.plot(x, df_plot["sma5"], label="SMA 5", color="gold", alpha=0.8, linewidth=1.5)
    ax.plot(x, df_plot["sma10"], label="SMA 10", color="dodgerblue", alpha=0.8, linewidth=1.5)
    ax.plot(x, df_plot["sma20"], label="SMA 20", color="magenta", alpha=0.8, linewidth=1.5)
    ax.plot(x, df_plot["sma50"], label="SMA 50", color="darkorange", alpha=0.8, linewidth=1.5)
    
    # Plot Candlesticks
    width = 0.6
    for i in range(len(df_plot)):
        color = "green" if o.iloc[i] < c.iloc[i] else "red"
        ax.vlines(x[i], l.iloc[i], h.iloc[i], color=color, linewidth=1)
        y0, height = min(o.iloc[i], c.iloc[i]), abs(c.iloc[i] - o.iloc[i])
        ax.add_patch(Rectangle((x[i] - width/2, y0), width, max(height, 0.001), facecolor=color, alpha=0.9))

    # Vertical Separator Line
    ax.axvline(split_index - 0.5, color="black", linestyle="--", linewidth=1.5, alpha=0.5)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc="upper left", fontsize=10)
    
    # X-axis formatting
    step = max(1, len(df_plot) // 12)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(df_plot["time"].dt.strftime("%m-%d %H:%M").iloc[::step], rotation=30)
    
    plt.grid(alpha=0.1)
    plt.tight_layout()
    return fig

# --- HELPER TO CONVERT IMAGE TO BASE64 ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# --- HEADER SECTION (FLEXBOX) ---
if os.path.exists("money_bag.png"):
    img_base64 = get_base64_of_bin_file("money_bag.png")
    header_html = f"""
        <div style="display: flex; align-items: center; gap: 30px; margin-bottom: 20px;">
            <img src="data:image/png;base64,{img_base64}" width="360">
            <h1 style="font-size: 32px; color: #FF4B4B; font-family: sans-serif; margin: 0;">SPX Trend</h1>
        </div>
    """
else:
    header_html = """
        <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 20px;">
            <span style="font-size: 80px;">💰</span>
            <h1 style="font-size: 32px; color: #FF4B4B; font-family: sans-serif; margin: 0;">SPX Trend</h1>
        </div>
    """

st.markdown(header_html, unsafe_allow_html=True)
st.markdown("Select the pattern of the first six 5-minute bars to identify historical market trends.")

# --- BIT INPUT TABLE ---
cols = st.columns(6)
times = ["08:30", "08:35", "08:40", "08:45", "08:50", "08:55"]
user_bits = []

for i, col in enumerate(cols):
    with col:
        choice = st.selectbox(
            f"{times[i]} Bar",
            options=["🔴 RED (0)", "🟢 GREEN (1)"],
            index=0,
            key=f"bar_{i}"
        )
        user_bits.append(0 if "RED" in choice else 1)

target_label = int(sum(w * b for w, b in zip(WEIGHTS, user_bits)))
bit_string = "".join(str(b) for b in user_bits)

st.info(f"**Pattern Sequence:** {bit_string} | **Label:** {target_label}")

# --- SEARCH AND DISPLAY ---
df = load_and_preprocess()

if df is not None:
    days = sorted(df["date"].unique())
    matches = []

    for i in range(1, len(days)):
        d_today, d_yday = days[i], days[i-1]
        today_data = df[df["date"] == d_today]
        day_label, day_bits = compute_label_from_df(today_data)
        
        if day_label == target_label:
            y_last = df[(df["date"] == d_yday) & (df["clock"] >= YDAY_LAST_HOUR_START) & (df["clock"] <= YDAY_LAST_HOUR_END)]
            t_sess = today_data[(today_data["clock"] >= TODAY_START) & (today_data["clock"] <= TODAY_END)]
            
            if not y_last.empty and not t_sess.empty:
                matches.append({
                    "date": d_today,
                    "bits": day_bits,
                    "plot_data": pd.concat([y_last, t_sess]),
                    "split": len(y_last)
                })

    st.divider()
    st.subheader(f"Found {len(matches)} historical matches for this pattern")

    for m in matches:
        with st.container():
            st.write(f"### Date: {m['date']}")
            fig = plot_candles_st(m['plot_data'], m['split'], f"SPX | {m['date']} | Pattern {bit_string}")
            st.pyplot(fig)
            st.divider()