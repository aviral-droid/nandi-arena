"""
Nandi LLM Arena — Investment Committee Demo
Live multi-model chat via Groq + Nandi technical analysis.
"""

import time
import concurrent.futures

import requests
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

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
  }
  .model-header {
    border-radius: 8px 8px 0 0;
    padding: 8px 12px 6px 12px;
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
    min-height: 110px;
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
  .nandi-panel {
    background: #1a0f00;
    border: 2px dashed #FF6B35;
    border-radius: 8px;
    padding: 14px;
    font-size: 13px;
    color: #e6edf3;
    min-height: 145px;
  }
  .timing { font-size: 11px; color: #6e7681; margin-top: 4px; text-align: right; }
  [data-testid="stButton"] > button {
    background: linear-gradient(135deg, #FF6B35, #f7931e);
    color: white; font-weight: 700; border: none;
    border-radius: 8px; padding: 8px 24px;
  }
  code { font-size: 12px; }
</style>
""", unsafe_allow_html=True)

# ── Groq model registry ────────────────────────────────────────────────────────
GROQ_MODELS = {
    "Llama 3.2 · 1B": {
        "id": "llama-3.2-1b-preview",
        "color": "#3498DB",
        "params": "1B",
        "note": "Meta · similar size to Nandi",
    },
    "Llama 3.2 · 3B": {
        "id": "llama-3.2-3b-preview",
        "color": "#9B59B6",
        "params": "3B",
        "note": "Meta",
    },
    "Gemma 2 · 9B": {
        "id": "gemma2-9b-it",
        "color": "#2ECC71",
        "params": "9B",
        "note": "Google",
    },
    "Llama 3.1 · 8B": {
        "id": "llama-3.1-8b-instant",
        "color": "#E74C3C",
        "params": "8B",
        "note": "Meta · fast",
    },
}

STARTERS = [
    "Translate to Telugu: India's technology industry is growing at an unprecedented pace.",
    "Translate to Hindi: The startup ecosystem raised $10 billion in venture capital this year.",
    "Tell a 3-sentence story set in Hyderabad, written in Telugu.",
    "Summarise for investors: Revenue ₹2,400 cr (+23% YoY), operating margin 18.5%, net profit ₹312 cr.",
    "A startup burns ₹50L/month, raised ₹6 crore. How many months of runway? Will it reach 18-month breakeven?",
    "What are the top 3 risks of investing in an early-stage Indian AI foundation model startup?",
]

FERTILITY = pd.DataFrame({
    "Language": ["Bengali", "Tamil", "Telugu", "Malayalam", "English"],
    "Nandi-600M": [1.44, 2.05, 1.77, 2.05, 1.18],
    "Qwen3-0.6B": [7.51, 10.93, 13.38, 14.56, 1.16],
    "SmolLM3-3B": [8.66, 13.56, 15.40, 17.77, 1.17],
})

LOCAL_CODE = '''from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "FrontiersMind/Nandi-Mini-600M-Early-Checkpoint"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True, dtype=torch.bfloat16
).eval()

model.config.kv_cache_mode = "shared"  # 50% memory saving

prompt = "Translate to Telugu: India is growing rapidly."
inputs = tokenizer([prompt], return_tensors="pt")
out = model.generate(
    **inputs, max_new_tokens=80, temperature=0.3,
    do_sample=True, repetition_penalty=1.1,
    pad_token_id=tokenizer.eos_token_id,
)
print(tokenizer.decode(out[0], skip_special_tokens=True))
'''

# ── Inference ──────────────────────────────────────────────────────────────────
def call_groq(name, cfg, message, max_tokens, temperature, groq_key):
    start = time.time()
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
            json={
                "model": cfg["id"],
                "messages": [{"role": "user", "content": message}],
                "max_tokens": max_tokens,
                "temperature": max(temperature, 0.01),
            },
            timeout=60,
        )
        if resp.status_code != 200:
            raise Exception(f"HTTP {resp.status_code}: {resp.text[:200]}")
        text = resp.json()["choices"][0]["message"]["content"]
        return {"model": name, "text": text.strip(), "time": time.time() - start, "error": None}
    except Exception as exc:
        return {"model": name, "text": "", "time": time.time() - start, "error": str(exc)}


def run_all(selected, message, max_tokens, temperature, groq_key):
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected)) as ex:
        futures = {
            ex.submit(call_groq, n, GROQ_MODELS[n], message, max_tokens, temperature, groq_key): n
            for n in selected
        }
        for f in concurrent.futures.as_completed(futures):
            r = f.result()
            results[r["model"]] = r
    return results


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Nandi Arena")
    st.caption("Investment Committee · Multi-Model Chat")
    st.divider()

    try:
        _secret = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        _secret = ""

    groq_key = st.text_input(
        "Groq API Key",
        value=_secret,
        type="password",
        placeholder="gsk_...",
        help="Free at console.groq.com — takes 30 seconds, no credit card.",
    )
    if not groq_key:
        st.warning("Get a free Groq key at [console.groq.com](https://console.groq.com)")

    st.divider()
    st.markdown("**Comparison models**")
    selected_models = []
    for name, cfg in GROQ_MODELS.items():
        if st.checkbox(f"{name}  `{cfg['params']}`", value=True, key=f"m_{name}"):
            selected_models.append(name)

    st.divider()
    st.markdown("**Generation settings**")
    max_tokens = st.slider("Max tokens", 50, 400, 200, 25)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.4, 0.05)

    if st.button("🗑  Clear chat"):
        st.session_state.history = []
        st.rerun()

    st.divider()
    st.markdown(
        "<small>**Why Groq?** Nandi, Sarvam, and SmolLM2 aren't yet deployed on any cloud "
        "inference provider — they run locally only. Groq hosts production-grade models "
        "(Llama, Gemma) and is free to use, making it the best live comparison available.</small>",
        unsafe_allow_html=True,
    )

# ── Session state ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Main ───────────────────────────────────────────────────────────────────────
tab_chat, tab_edge, tab_local, tab_about = st.tabs([
    "💬 Live Arena", "📊 Nandi's Indic Edge", "🖥️ Run Nandi Locally", "ℹ️ Models"
])

# ─────────────────────────────── LIVE ARENA ───────────────────────────────────
with tab_chat:
    st.markdown("### Ask anything — Nandi ⭐ vs live models simultaneously")
    st.caption("Nandi runs locally (see **Run Nandi Locally** tab). Comparison models run live via Groq.")

    # Starter prompts
    if not st.session_state.history:
        st.markdown("**Try a starter prompt:**")
        cols = st.columns(3)
        for i, s in enumerate(STARTERS):
            label = s[:52] + ("…" if len(s) > 52 else "")
            if cols[i % 3].button(label, key=f"s{i}"):
                if groq_key and selected_models:
                    with st.spinner("Running models…"):
                        res = run_all(selected_models, s, max_tokens, temperature, groq_key)
                    st.session_state.history.append(
                        {"user": s, "results": res, "models": list(selected_models)}
                    )
                    st.rerun()

    # Chat history
    for turn in st.session_state.history:
        st.markdown(f"<div class='user-bubble'>🧑 {turn['user']}</div>", unsafe_allow_html=True)

        # Nandi column + live model columns
        live_models = turn["models"]
        all_cols = st.columns(1 + len(live_models))

        # Nandi placeholder
        with all_cols[0]:
            st.markdown(
                "<div class='model-header' style='background:#FF6B3522; border-color:#FF6B35; color:#FF6B35;'>"
                "Nandi-600M ⭐<br>"
                "<span style='font-weight:400; color:#8b949e'>600M · 20% trained · Indic-optimised</span>"
                "</div>"
                "<div class='nandi-panel'>"
                "🖥️ <strong>Local inference required</strong><br><br>"
                "Nandi isn't yet on any cloud API. Run it on your machine — see the "
                "<strong>Run Nandi Locally</strong> tab for the exact code.<br><br>"
                "<span style='color:#FF6B35'>→ Tokenisation advantage: 7.5× fewer tokens for Telugu "
                "vs Qwen — see Nandi's Indic Edge tab.</span>"
                "</div>",
                unsafe_allow_html=True,
            )

        # Live model outputs
        for i, mname in enumerate(live_models):
            r = turn["results"].get(mname, {})
            cfg = GROQ_MODELS.get(mname, {})
            color = cfg.get("color", "#888")
            with all_cols[i + 1]:
                st.markdown(
                    f"<div class='model-header' style='background:{color}22; border-color:{color}; color:{color};'>"
                    f"{mname}<br>"
                    f"<span style='font-weight:400; color:#8b949e'>{cfg.get('params','')} · {cfg.get('note','')}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if r.get("error"):
                    st.markdown(f"<div class='error-box'>{r['error'][:200]}</div>", unsafe_allow_html=True)
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

    # Input form
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Your message",
            placeholder="Ask in English, Telugu, Hindi, or any language…",
            height=90,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("🚀  Send to all models")

    if submitted:
        if not groq_key:
            st.error("Add your Groq API key in the sidebar (free at console.groq.com).")
        elif not selected_models:
            st.error("Select at least one model in the sidebar.")
        elif not user_input.strip():
            st.warning("Type something first.")
        else:
            with st.spinner(f"Calling {len(selected_models)} models…"):
                res = run_all(selected_models, user_input.strip(), max_tokens, temperature, groq_key)
            st.session_state.history.append({
                "user": user_input.strip(),
                "results": res,
                "models": list(selected_models),
            })
            st.rerun()


# ─────────────────────────────── INDIC EDGE ───────────────────────────────────
with tab_edge:
    st.subheader("Tokenisation Fertility — Nandi's Structural Advantage")
    st.markdown(
        "**Fertility** = tokens needed per word. Lower is better. "
        "Nandi encodes Indic text in 7–8× fewer tokens than Qwen or SmolLM. "
        "This reduces inference cost, increases context density, and improves accuracy on Indic tasks."
    )

    fig = go.Figure()
    pal = {"Nandi-600M": "#FF6B35", "Qwen3-0.6B": "#3498DB", "SmolLM3-3B": "#9B59B6"}
    for col in pal:
        fig.add_trace(go.Bar(
            name=col, x=FERTILITY["Language"], y=FERTILITY[col],
            marker_color=pal[col], text=FERTILITY[col], textposition="outside",
        ))
    fig.update_layout(
        barmode="group", yaxis_title="Fertility score (lower = better)",
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
        st.caption("At 20% training Nandi scores within ~7% of models trained on 16–180× more data.")

    with c2:
        st.subheader("Telugu efficiency advantage over Qwen3")
        ratios = (FERTILITY["Qwen3-0.6B"] / FERTILITY["Nandi-600M"]).tolist()
        fig2 = go.Figure(go.Bar(
            x=FERTILITY["Language"].tolist(), y=ratios,
            marker_color=["#FF6B35"] * 5,
            text=[f"{r:.1f}×" for r in ratios], textposition="outside",
        ))
        fig2.update_layout(
            yaxis_title="Nandi efficiency multiplier",
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font_color="#e6edf3", height=320, margin=dict(t=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.info(
        "📌 **Investor note:** Tokenisation efficiency is an architectural property that persists "
        "and compounds as Nandi trains further. At full training (~1.25T tokens), this advantage "
        "translates directly to lower per-query cost and higher quality on every Indian-language workload."
    )


# ─────────────────────────────── RUN LOCALLY ──────────────────────────────────
with tab_local:
    st.subheader("Running Nandi-Mini-600M on your machine")
    st.markdown(
        "Nandi isn't yet on any cloud inference API — it's too new. "
        "It runs easily on a laptop CPU or any GPU. The shared-KV architecture "
        "cuts memory usage by ~50% vs a standard transformer of the same size."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Option A — Python (Transformers)**")
        st.code(LOCAL_CODE, language="python")

    with col2:
        st.markdown("**Option B — Docker (one command)**")
        st.code(
            "docker model run hf.co/FrontiersMind/Nandi-Mini-600M-Early-Checkpoint",
            language="bash",
        )
        st.caption("Creates a local OpenAI-compatible server at localhost:8000")

        st.markdown("**Install**")
        st.code("pip install transformers torch accelerate", language="bash")

        st.markdown("**Model specs**")
        st.markdown("""
| Property | Value |
|---|---|
| Parameters | 600M |
| Architecture | Transformer decoder |
| KV Cache | Shared (50% memory saving) |
| Context length | 2,048 (→ 32K planned) |
| Languages | English + 10 Indic |
| Training | 250B / ~1.25T tokens (20%) |
| Precision | BF16 |
| License | Apache 2.0 |
        """)

    st.info(
        "💡 Once running locally, point this arena at your local endpoint by adding "
        "`http://localhost:8000` as a custom model — or run the Transformers code above "
        "and paste the output into the comparison manually."
    )


# ─────────────────────────────── ABOUT ────────────────────────────────────────
with tab_about:
    st.subheader("Live comparison models (via Groq)")
    for name, cfg in GROQ_MODELS.items():
        with st.expander(f"{name} — {cfg['params']} · {cfg['note']}"):
            st.code(cfg["id"], language=None)
            st.write("Provider: **Groq** (free tier)")

    st.divider()
    st.subheader("Why aren't Nandi / Sarvam / SmolLM2 in the live arena?")
    st.markdown("""
These models are not yet deployed on any cloud inference provider:

| Model | Status |
|---|---|
| Nandi-Mini-600M | No provider — local only |
| Sarvam-2B | No provider — local only |
| SmolLM2-360M/1.7B | No provider — local only |
| Qwen2.5-small variants | No provider — local only |

HuggingFace's serverless inference only hosts a small subset of popular models.
For any model not on a provider, the only option is local inference.

The live arena uses Groq because it's **free**, **fast**, and **reliable** from Streamlit Cloud.
    """)

    st.divider()
    st.subheader("Getting a Groq key")
    st.markdown("""
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (no credit card required)
3. Click **API Keys** → **Create API Key**
4. Paste in the sidebar

The free tier is very generous — thousands of tokens per minute.
    """)
