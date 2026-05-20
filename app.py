"""
Nandi LLM Arena — Investment Committee Multi-Model Chatbot
Type once, see all model responses side-by-side in real time.
"""

import time
import concurrent.futures

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from huggingface_hub import InferenceClient

st.set_page_config(
    page_title="Nandi LLM Arena",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .user-bubble {
    background: #1f3a5c;
    border-radius: 12px 12px 4px 12px;
    padding: 12px 16px;
    margin: 16px 0 8px 0;
    font-size: 15px;
    color: #e6edf3;
    display: inline-block;
    max-width: 100%;
  }
  .model-header {
    border-radius: 8px 8px 0 0;
    padding: 8px 12px;
    font-size: 12px;
    font-weight: 700;
    border-left: 4px solid;
  }
  .model-output {
    background: #0d1117;
    border: 1px solid #30363d;
    border-top: none;
    border-radius: 0 0 8px 8px;
    padding: 12px 14px;
    min-height: 100px;
    font-size: 13px;
    line-height: 1.8;
    white-space: pre-wrap;
    color: #e6edf3;
    font-family: 'Segoe UI', system-ui, sans-serif;
  }
  .error-box {
    background: #2d1a1a;
    border: 1px solid #7f1d1d;
    border-top: none;
    border-radius: 0 0 8px 8px;
    padding: 10px 14px;
    font-size: 12px;
    color: #fca5a5;
  }
  .timing { font-size: 11px; color: #6e7681; margin-top: 4px; text-align: right; }
  .starter-chip {
    display: inline-block;
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 13px;
    color: #c9d1d9;
    cursor: pointer;
    margin: 4px;
  }
  [data-testid="stButton"] > button {
    background: linear-gradient(135deg, #FF6B35, #f7931e);
    color: white; font-weight: 700; border: none;
    border-radius: 8px; padding: 8px 24px;
  }
</style>
""", unsafe_allow_html=True)

# ── Models ─────────────────────────────────────────────────────────────────────
MODELS = {
    "Nandi-600M ⭐": {
        "id": "FrontiersMind/Nandi-Mini-600M-Early-Checkpoint",
        "color": "#FF6B35", "type": "base",
        "params": "600M", "note": "20% trained • Indic-optimised", "highlight": True,
    },
    "Sarvam-2B": {
        "id": "sarvamai/sarvam-2b-v0.5",
        "color": "#F39C12", "type": "instruct",
        "params": "2B", "note": "Indic specialist • Sarvam AI", "highlight": False,
    },
    "SmolLM2-360M": {
        "id": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "color": "#2ECC71", "type": "instruct",
        "params": "360M", "note": "HuggingFace", "highlight": False,
    },
    "SmolLM2-1.7B": {
        "id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "color": "#9B59B6", "type": "instruct",
        "params": "1.7B", "note": "HuggingFace", "highlight": False,
    },
    "Qwen2.5-0.5B": {
        "id": "Qwen/Qwen2.5-0.5B-Instruct",
        "color": "#3498DB", "type": "instruct",
        "params": "500M", "note": "Alibaba", "highlight": False,
    },
    "Qwen2.5-1.5B": {
        "id": "Qwen/Qwen2.5-1.5B-Instruct",
        "color": "#E74C3C", "type": "instruct",
        "params": "1.5B", "note": "Alibaba", "highlight": False,
    },
}

STARTERS = [
    "Translate to Telugu: India's technology industry is transforming rapidly.",
    "Translate to Hindi: The startup ecosystem raised $10 billion this year.",
    "Tell me a short story set in Hyderabad (in Telugu).",
    "Summarise in 3 bullets: Revenue ₹2,400 cr (+23% YoY), margin 18.5%, net profit ₹312 cr.",
    "A startup burns ₹50L/month and has ₹6 cr raised. When does it run out of runway?",
    "What are the top 3 risks of investing in early-stage Indian AI startups?",
]

FERTILITY = pd.DataFrame({
    "Language": ["Bengali", "Tamil", "Telugu", "Malayalam", "English"],
    "Nandi-600M": [1.44, 2.05, 1.77, 2.05, 1.18],
    "Qwen3-0.6B": [7.51, 10.93, 13.38, 14.56, 1.16],
    "SmolLM3-3B": [8.66, 13.56, 15.40, 17.77, 1.17],
})

# ── Inference ──────────────────────────────────────────────────────────────────
def call_model(name, cfg, message, max_tokens, temperature, hf_token):
    start = time.time()
    try:
        client = InferenceClient(provider="hf-inference", api_key=hf_token)

        if cfg["type"] == "base":
            text = client.text_generation(
                model=cfg["id"],
                prompt=message,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                repetition_penalty=1.1,
                do_sample=True,
            )
        else:
            resp = client.chat_completion(
                model=cfg["id"],
                messages=[{"role": "user", "content": message}],
                max_tokens=max_tokens,
                temperature=max(temperature, 0.01),
            )
            text = resp.choices[0].message.content or ""

        return {"model": name, "text": text.strip(), "time": time.time() - start, "error": None}
    except Exception as exc:
        return {"model": name, "text": "", "time": time.time() - start, "error": str(exc)}


def run_all(selected, message, max_tokens, temperature, hf_token):
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected)) as ex:
        futures = {
            ex.submit(call_model, n, MODELS[n], message, max_tokens, temperature, hf_token): n
            for n in selected
        }
        for f in concurrent.futures.as_completed(futures):
            r = f.result()
            results[r["model"]] = r
    return results


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Nandi Arena")
    st.caption("Investment Committee · Multi-Model Chatbot")
    st.divider()

    try:
        _secret = st.secrets.get("HF_TOKEN", "")
    except Exception:
        _secret = ""

    hf_token = st.text_input(
        "HuggingFace Token", value=_secret, type="password",
        placeholder="hf_...",
        help="Free read token from huggingface.co/settings/tokens",
    )
    if not hf_token:
        st.warning("Add your HF token to start.")

    st.divider()
    st.markdown("**Select models**")
    selected_models = []
    for name, cfg in MODELS.items():
        star = " ⭐" if cfg["highlight"] else ""
        if st.checkbox(f"{name}  `{cfg['params']}`", value=True, key=f"m_{name}"):
            selected_models.append(name)

    st.divider()
    st.markdown("**Settings**")
    max_tokens = st.slider("Max tokens", 50, 400, 200, 25)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.4, 0.05)

    if st.button("🗑  Clear chat"):
        st.session_state.history = []
        st.rerun()


# ── Session state ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Main ───────────────────────────────────────────────────────────────────────
tab_chat, tab_edge, tab_about = st.tabs(["💬 Chat Arena", "📊 Nandi's Indic Edge", "ℹ️ Models"])

with tab_chat:
    st.markdown("### Ask anything — see all models respond simultaneously")

    # Starter prompts
    if not st.session_state.history:
        st.markdown("**Try a starter prompt:**")
        cols = st.columns(3)
        for i, s in enumerate(STARTERS):
            if cols[i % 3].button(s[:55] + ("…" if len(s) > 55 else ""), key=f"s{i}"):
                if hf_token and selected_models:
                    with st.spinner("Running models…"):
                        res = run_all(selected_models, s, max_tokens, temperature, hf_token)
                    st.session_state.history.append({"user": s, "results": res, "models": selected_models})
                    st.rerun()

    # Chat history
    for turn in st.session_state.history:
        st.markdown(
            f"<div class='user-bubble'>🧑 {turn['user']}</div>",
            unsafe_allow_html=True,
        )
        n = len(turn["models"])
        cols = st.columns(n)
        for i, mname in enumerate(turn["models"]):
            r = turn["results"].get(mname, {})
            cfg = MODELS.get(mname, {})
            color = cfg.get("color", "#888")
            with cols[i]:
                st.markdown(
                    f"<div class='model-header' style='background:{color}22; border-color:{color}; color:{color};'>"
                    f"{mname}<br><span style='font-weight:400; color:#8b949e'>{cfg.get('params','')} · {cfg.get('note','')}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if r.get("error"):
                    st.markdown(
                        f"<div class='error-box'>{r['error'][:250]}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<div class='model-output'>{r.get('text','') or '(no output)'}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='timing'>⏱ {r.get('time', 0):.2f}s</div>",
                        unsafe_allow_html=True,
                    )
        st.divider()

    # Input
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Your message",
            placeholder="Ask in English, Telugu, Hindi, or any language…",
            height=90,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("🚀  Send to all models")

    if submitted:
        if not hf_token:
            st.error("Add your HuggingFace token in the sidebar first.")
        elif not selected_models:
            st.error("Select at least one model in the sidebar.")
        elif not user_input.strip():
            st.warning("Type something first.")
        else:
            with st.spinner(f"Calling {len(selected_models)} models…"):
                res = run_all(selected_models, user_input.strip(), max_tokens, temperature, hf_token)
            st.session_state.history.append({
                "user": user_input.strip(),
                "results": res,
                "models": selected_models,
            })
            st.rerun()


# ── Indic Edge tab ─────────────────────────────────────────────────────────────
with tab_edge:
    st.subheader("Tokenization Fertility — Nandi's Structural Advantage")
    st.markdown(
        "**Fertility** = tokens per word. Lower is better: Nandi encodes Indic text "
        "in far fewer tokens → lower inference cost, better context retention, higher accuracy."
    )

    fig = go.Figure()
    pal = {"Nandi-600M": "#FF6B35", "Qwen3-0.6B": "#3498DB", "SmolLM3-3B": "#9B59B6"}
    for col in pal:
        fig.add_trace(go.Bar(
            name=col, x=FERTILITY["Language"], y=FERTILITY[col],
            marker_color=pal[col], text=FERTILITY[col], textposition="outside",
        ))
    fig.update_layout(
        barmode="group", yaxis_title="Fertility (lower = better)",
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font_color="#e6edf3", height=380, legend=dict(bgcolor="#0d1117"),
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Benchmark scores  (* early checkpoint)")
        st.dataframe(pd.DataFrame({
            "Model": ["Nandi-600M *", "SmolLM2-360M", "Qwen3-0.6B"],
            "Tokens Trained": ["250B (20%)", "4T", "36T"],
            "HellaSwag": [44.86, 56.30, 53.77],
            "WinoGrande": [54.77, 59.19, 59.19],
            "MMLU": [29.01, 25.55, 50.34],
            "Average": [44.10, 47.53, 49.75],
        }), use_container_width=True, hide_index=True)
        st.caption("At 20% training, Nandi already approaches models trained on 16–180× more data.")

    with c2:
        st.subheader("Telugu efficiency ratio vs Qwen3")
        ratios = (FERTILITY["Qwen3-0.6B"] / FERTILITY["Nandi-600M"]).tolist()
        fig2 = go.Figure(go.Bar(
            x=FERTILITY["Language"].tolist(), y=ratios,
            marker_color=["#FF6B35"] * 5,
            text=[f"{r:.1f}×" for r in ratios], textposition="outside",
        ))
        fig2.update_layout(
            yaxis_title="Nandi efficiency advantage",
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font_color="#e6edf3", height=320, margin=dict(t=20),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.info(
        "📌 **Investor note:** Nandi's tokenisation advantage is architectural — "
        "it persists and compounds as training scales. A 7.5× token reduction for Telugu "
        "means 7.5× more context in the same window and proportionally lower API costs "
        "for every Indian-language query."
    )


# ── About tab ──────────────────────────────────────────────────────────────────
with tab_about:
    st.subheader("Models in this arena")
    for name, cfg in MODELS.items():
        with st.expander(f"{name}  —  {cfg['params']}  ·  {cfg['note']}"):
            st.code(cfg["id"], language=None)
            st.write(f"**Type:** {cfg['type'].capitalize()}")
            if cfg["highlight"]:
                st.info(
                    "Early pretraining checkpoint (250B / ~1.25T planned tokens). "
                    "Architecture: Shared KV (50% memory saving), GQA, SwiGLU, RoPE. "
                    "Supports English + 10 Indic languages."
                )
    st.divider()
    st.markdown("""
**How inference works**

All models are called simultaneously via HuggingFace's own inference infrastructure
(`provider="hf-inference"`). Base models (Nandi) receive a completion-style prompt;
instruct models receive a chat message. Results race back and display as they arrive.

**Token** — get a free read token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
    """)
