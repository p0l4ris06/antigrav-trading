"""
THE ZENITH TERMINAL: APEX COMMAND
==================================
Final Evolutionary State.
Shadow Brain Monitoring / Dual-Fork Telemetry.
"""

import streamlit as st
import json
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
import time
import subprocess
from datetime import datetime
from dotenv import load_dotenv

# 1. DESIGN SYSTEM
st.set_page_config(page_title="ZENITH APEX", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;500;700&family=JetBrains+Mono&display=swap');
    :root { --apex-cyan: #00f2ff; --apex-purple: #7000ff; --apex-bg: #010103; }
    html, body, [data-testid="stAppViewContainer"] { background-color: var(--apex-bg); font-family: 'Space Grotesk', sans-serif; color: #e0e0e0; }
    .apex-card { background: rgba(15, 15, 25, 0.9); border: 1px solid rgba(112, 0, 255, 0.2); border-radius: 12px; padding: 20px; backdrop-filter: blur(20px); margin-bottom: 15px; }
    .glow-text { color: var(--apex-cyan); text-shadow: 0 0 10px rgba(0, 242, 255, 0.3); font-family: 'JetBrains Mono'; }
    .matrix-log { background: #000; color: var(--apex-cyan); font-family: 'JetBrains Mono'; font-size: 0.75rem; padding: 15px; border: 1px solid rgba(112, 0, 255, 0.2); height: 350px; overflow-y: scroll; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# 2. DATA ORCHESTRATOR
def get_state():
    try:
        if os.path.exists("bot_state.json"):
            with open("bot_state.json", "r") as f: 
                data = json.load(f)
                if data: st.session_state['last_state'] = data
                return data
    except: pass
    return st.session_state.get('last_state', None)

state = get_state()

# 3. HEADER
h1, h2, h3 = st.columns([3, 1, 1])
h1.markdown("<h1 style='font-size:4rem; margin:0; line-height:1; letter-spacing:-3px;'>ZENITH <span style='color:#7000ff; font-weight:200;'>APEX</span></h1>", unsafe_allow_html=True)
with h2:
    if st.button("⚡ INITIALIZE", use_container_width=True):
        subprocess.Popen(["python", "bot.py"], creationflags=subprocess.CREATE_NO_WINDOW)
        st.toast("Zenith Apex Engine Online")
with h3:
    if st.button("🛑 KILL-SWITCH", use_container_width=True):
        os.system("taskkill /F /IM python.exe /T")
        st.toast("Emergency Halt Executed")

# 4. COMMAND CENTER
if state:
    # SHADOW BRAIN TELEMETRY
    weights = state.get("shadow_weights", {"trend": 0, "rsi": 0})
    s1, s2, s3, s4 = st.columns(4)
    with s1: st.markdown(f"""<div class="apex-card"><div style="font-size:0.7rem; color:#666;">REGIME SENSOR</div><div class="glow-text">{state.get('regime')}</div></div>""", unsafe_allow_html=True)
    with s2: st.markdown(f"""<div class="apex-card"><div style="font-size:0.7rem; color:#666;">SHADOW TREND WEIGHT</div><div class="glow-text">{weights['trend']:.2f}</div></div>""", unsafe_allow_html=True)
    with s3: st.markdown(f"""<div class="apex-card"><div style="font-size:0.7rem; color:#666;">FORK B STATUS</div><div class="glow-text" style="color:#00ff88">LOGGING DEPTH</div></div>""", unsafe_allow_html=True)
    with s4: st.markdown(f"""<div class="apex-card"><div style="font-size:0.7rem; color:#666;">SYSTEM HEALTH</div><div class="glow-text">OPTIMAL</div></div>""", unsafe_allow_html=True)

    # MAIN VISUALIZER
    c1, c2 = st.columns([1, 2.5])
    with c1:
        st.markdown("<div class='apex-card'>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.7rem; color:#666; letter-spacing:2px; margin-bottom:15px;'>GLOBAL PULSE</p>", unsafe_allow_html=True)
        for ticker, data in state.get("ohlc", {}).items():
            st.markdown(f"""<div style="padding:10px; border-bottom:1px solid rgba(112,0,255,0.05)"><div style="font-size:0.6rem; color:#444;">{ticker}</div><div style="font-size:1rem; font-weight:700;">${data[-1][-1]:.2f}</div></div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='apex-card'>", unsafe_allow_html=True)
        best_key = state.get("best_pick_key")
        if best_key and best_key in state.get("ohlc", {}):
            df = pd.DataFrame(state["ohlc"][best_key], columns=['Date', 'Open', 'High', 'Low', 'Close'])
            df['Date'] = pd.to_datetime(df['Date'])
            fig = sp.make_subplots(rows=1, cols=1)
            fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                            increasing_line_color='#00f2ff', decreasing_line_color='#7000ff', name="Live"))
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_rangeslider_visible=False, height=500, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)

    # LOGS
    st.markdown("<div class='apex-card'>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.7rem; color:#666; letter-spacing:2px; margin-bottom:10px;'>ZENITH APEX AUDIT</p>", unsafe_allow_html=True)
    if os.path.exists("trading_log.txt"):
        with open("trading_log.txt", "r") as f:
            logs = "".join(f.readlines()[-20:])
            st.markdown(f"<pre class='matrix-log'>{logs}</pre>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.warning("ZENITH APEX OFFLINE // INITIATE HANDSHAKE")

time.sleep(1)
st.rerun()
