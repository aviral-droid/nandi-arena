"""
Nandi LLM Arena — Investment Committee Demo
Live multi-model chat · Nandi via local endpoint · Live metrics comparison
"""

import time
import concurrent.futures

import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
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
    background: #1a2d45;
    border-radius: 10px;
    padding: 10px 16px;
    margin: 14px 0 6px 0;
    font-size: 14px;
    color: #c9d1d9;
  }
  .model-header {
    border-radius: 8px 8px 0 0;
    padding: 8px 12px 5px 12px;
    font-size: 11px;
    font-weight: 700;
    border-left: 4px solid;
    letter-spacing: .3px;
  }
  .model-output {
    background: #0d1117;
    border: 1px solid #30363d;
    border-top: none;
    border-radius: 0 0 8px 8px;
    padding: 11px 13px;
    min-height: 100px;
    font-size: 13px;
    line-height: 1.75;
    white-space: pre-wrap;
    color: #e6edf3;
    font-family: 'Segoe UI', system-ui, sans-serif;
  }
  .error-box {
    background: #2d1a1a;
    border: 1px solid #7f1d1d;
    border-top: none;
    border-radius: 0 0 8px 8px;
    padding: 9px 13px;
    font-size: 11px;
    color: #fca5a5;
  }
  .nandi-local {
    background: #1a0f00;
    border: 2px dashed #FF6B35;
    border-radius: 8px;
    padding: 12px;
    font-size: 12px;
    color: #8b949e;
    min-height: 100px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
  }
  .timing { font-size: 10px; color: #6e7681; margin-top: 3px; text-align: right; }
  .metric-badge {
    display: inline-block;
    border-radius: 4px;
    padding: 2px 7px;
    font-size: 11px;
    font-weight: 600;
    margin: 1px 0;
  }
  [data-testid="stButton"] > button {
    background: linear-gradient(135deg, #FF6B35, #f7931e);
    color: white; font-weight: 700; border: none;
    border-radius: 8px; padding: 7px 20px;
  }
</style>
""", unsafe_allow_html=True)

# ── Model registry ─────────────────────────────────────────────────────────────
MODELS = {
    "Sarvam-30B 🇮🇳": {
        "id": "sarvam-30b",
        "provider": "sarvam",
        "color": "#F39C12",
        "params": "30B",
        "note": "Indic specialist · Sarvam AI",
        "size_b": 30,
    },
    "Llama 3.1 · 8B": {
        "id": "llama-3.1-8b-instant",
        "provider": "groq",
        "color": "#3498DB",
        "params": "8B",
        "note": "Meta · fast",
        "size_b": 8,
    },
    "Llama 3.3 · 70B": {
        "id": "llama-3.3-70b-versatile",
        "provider": "groq",
        "color": "#9B59B6",
        "params": "70B",
        "note": "Meta",
        "size_b": 70,
    },
    "GPT-OSS · 20B": {
        "id": "openai/gpt-oss-20b",
        "provider": "groq",
        "color": "#2ECC71",
        "params": "20B",
        "note": "OpenAI OSS · fast",
        "size_b": 20,
    },
    "GPT-OSS · 120B": {
        "id": "openai/gpt-oss-120b",
        "provider": "groq",
        "color": "#E74C3C",
        "params": "120B",
        "note": "OpenAI OSS · largest",
        "size_b": 120,
    },
}

PROVIDER_URLS = {
    "groq":   "https://api.groq.com/openai/v1/chat/completions",
    "sarvam": "https://api.sarvam.ai/v1/chat/completions",
}

STARTERS = [
    "Translate to Telugu: India's technology industry is growing at an unprecedented pace.",
    "Translate to Hindi: The startup ecosystem raised $10 billion in venture capital this year.",
    "Tell a 3-sentence story set in Hyderabad, written in Telugu.",
    "Summarise for investors: Revenue ₹2,400 cr (+23% YoY), margin 18.5%, net profit ₹312 cr.",
    "Startup burns ₹50L/month, raised ₹6 crore. Months of runway? Will it reach breakeven in 18 months?",
    "Top 3 risks of investing in an early-stage Indian AI foundation model startup?",
]

# Static reference data
FERTILITY = pd.DataFrame({
    "Language":    ["Bengali", "Tamil", "Telugu", "Malayalam", "English"],
    "Nandi-600M":  [1.44, 2.05, 1.77, 2.05, 1.18],
    "Qwen3-0.6B":  [7.51, 10.93, 13.38, 14.56, 1.16],
    "SmolLM3-3B":  [8.66, 13.56, 15.40, 17.77, 1.17],
})

BENCHMARKS = pd.DataFrame({
    "Model":           ["Nandi-600M *", "SmolLM2-360M", "Qwen3-0.6B"],
    "Tokens Trained":  ["250B (20%)",   "4T",           "36T"],
    "HellaSwag":       [44.86, 56.30, 53.77],
    "WinoGrande":      [54.77, 59.19, 59.19],
    "MMLU":            [29.01, 25.55, 50.34],
    "Average":         [44.10, 47.53, 49.75],
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
    **inputs, max_new_tokens=100, temperature=0.3, do_sample=True,
    repetition_penalty=1.1, pad_token_id=tokenizer.eos_token_id,
)
print(tokenizer.decode(out[0], skip_special_tokens=True))
'''


# ── Inference ──────────────────────────────────────────────────────────────────
def call_model(name, cfg, message, max_tokens, temperature, groq_key, sarvam_key):
    start = time.time()
    provider = cfg["provider"]
    api_key = sarvam_key if provider == "sarvam" else groq_key

    if not api_key:
        label = "Sarvam" if provider == "sarvam" else "Groq"
        return {"model": name, "text": "", "time": 0.0, "tokens": 0,
                "error": f"{label} API key not provided — add it in the sidebar."}
    try:
        resp = requests.post(
            PROVIDER_URLS[provider],
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": cfg["id"],
                "messages": [{"role": "user", "content": message}],
                "max_tokens": max_tokens,
                "temperature": max(temperature, 0.01),
            },
            timeout=60,
        )
        if resp.status_code != 200:
            raise Exception(f"HTTP {resp.status_code}: {resp.text[:300]}")
        body = resp.json()
        text   = body["choices"][0]["message"]["content"]
        tokens = body.get("usage", {}).get("completion_tokens", len(text.split()))
        return {"model": name, "text": text.strip(), "time": time.time() - start,
                "tokens": tokens, "error": None}
    except Exception as exc:
        return {"model": name, "text": "", "time": time.time() - start,
                "tokens": 0, "error": str(exc)}


def call_nandi(nandi_url, message, max_tokens, temperature):
    """Call a locally-hosted Nandi endpoint (Docker model runner / vLLM / SGLang)."""
    start = time.time()
    base = nandi_url.rstrip("/")
    # Try chat/completions first (Docker model runner, vLLM --chat)
    for path, payload in [
        ("/v1/chat/completions", {
            "model": "FrontiersMind/Nandi-Mini-600M-Early-Checkpoint",
            "messages": [{"role": "user", "content": message}],
            "max_tokens": max_tokens,
            "temperature": max(temperature, 0.01),
        }),
        ("/v1/completions", {
            "model": "FrontiersMind/Nandi-Mini-600M-Early-Checkpoint",
            "prompt": message,
            "max_tokens": max_tokens,
            "temperature": max(temperature, 0.01),
        }),
    ]:
        try:
            resp = requests.post(base + path, json=payload, timeout=120)
            if resp.status_code == 200:
                body = resp.json()
                if "choices" in body:
                    ch = body["choices"][0]
                    text = ch.get("message", {}).get("content") or ch.get("text", "")
                    tokens = body.get("usage", {}).get("completion_tokens", len(text.split()))
                    return {"model": "Nandi-600M ⭐", "text": text.strip(),
                            "time": time.time() - start, "tokens": tokens, "error": None}
        except requests.exceptions.ConnectionError:
            return {"model": "Nandi-600M ⭐", "text": "", "time": 0.0, "tokens": 0,
                    "error": f"Cannot connect to {base} — is Nandi running locally?"}
        except Exception:
            continue
    return {"model": "Nandi-600M ⭐", "text": "", "time": 0.0, "tokens": 0,
            "error": f"No valid response from {base}"}


def run_all(selected, message, max_tokens, temperature, groq_key, sarvam_key, nandi_url):
    results = {}
    tasks = {n: (call_model, (n, MODELS[n], message, max_tokens, temperature, groq_key, sarvam_key))
             for n in selected}
    if nandi_url:
        tasks["Nandi-600M ⭐"] = (call_nandi, (nandi_url, message, max_tokens, temperature))

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as ex:
        futures = {ex.submit(fn, *args): key for key, (fn, args) in tasks.items()}
        for f in concurrent.futures.as_completed(futures):
            r = f.result()
            results[futures[f]] = r
    return results


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Nandi Arena")
    st.caption("Investment Committee · Multi-Model Chat")
    st.divider()

    try:
        _groq_s   = st.secrets.get("GROQ_API_KEY",   "")
        _sarvam_s = st.secrets.get("SARVAM_API_KEY", "")
        _nandi_s  = st.secrets.get("NANDI_URL",      "")
    except Exception:
        _groq_s = _sarvam_s = _nandi_s = ""

    groq_key = st.text_input(
        "Groq API Key", value=_groq_s, type="password", placeholder="gsk_...",
        help="Free at console.groq.com",
    )
    sarvam_key = st.text_input(
        "Sarvam API Key", value=_sarvam_s, type="password", placeholder="...",
        help="Get at dashboard.sarvam.ai",
    )

    st.divider()
    st.markdown("**🟠 Connect Nandi locally**")
    nandi_url = st.text_input(
        "Nandi endpoint URL", value=_nandi_s,
        placeholder="http://localhost:8080",
        help=(
            "Run Nandi locally then paste the base URL here.\n\n"
            "Docker: docker model run hf.co/FrontiersMind/Nandi-Mini-600M-Early-Checkpoint\n"
            "vLLM:   vllm serve FrontiersMind/Nandi-Mini-600M-Early-Checkpoint\n\n"
            "Base URL only — e.g. http://localhost:8080"
        ),
    )
    if nandi_url:
        st.success("Nandi will run live in the arena!")
    else:
        st.caption("No URL → Nandi shown as local-only placeholder")

    st.divider()
    st.markdown("**Models**")
    selected_models = []
    for name, cfg in MODELS.items():
        if st.checkbox(f"{name}  `{cfg['params']}`", value=True, key=f"m_{name}"):
            selected_models.append(name)

    st.divider()
    st.markdown("**Generation settings**")
    max_tokens  = st.slider("Max tokens",  50, 400, 200, 25)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.4, 0.05)

    if st.button("🗑  Clear chat"):
        st.session_state.history = []
        st.rerun()


# ── Session state ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Main ───────────────────────────────────────────────────────────────────────
tab_chat, tab_metrics, tab_edge, tab_local, tab_about = st.tabs([
    "💬 Live Arena", "📈 Live Metrics", "📊 Nandi's Indic Edge",
    "🖥️ Run Nandi Locally", "ℹ️ Models",
])


# ═══════════════════════════════ LIVE ARENA ═══════════════════════════════════
with tab_chat:
    nandi_live = bool(nandi_url)
    total_models = (1 if nandi_live else 1) + len(selected_models)

    if nandi_live:
        st.success("Nandi is connected — running live in all queries.")
    else:
        st.info("Nandi not connected. Provide a local endpoint URL in the sidebar to run it live.")

    # Starter prompts
    if not st.session_state.history:
        st.markdown("**Try a starter:**")
        cols = st.columns(3)
        for i, s in enumerate(STARTERS):
            label = s[:50] + ("…" if len(s) > 50 else "")
            if cols[i % 3].button(label, key=f"s{i}"):
                if groq_key or nandi_url:
                    with st.spinner("Running all models simultaneously…"):
                        res = run_all(selected_models, s, max_tokens, temperature,
                                      groq_key, sarvam_key, nandi_url)
                    st.session_state.history.append(
                        {"user": s, "results": res, "models": list(selected_models),
                         "nandi_live": nandi_live}
                    )
                    st.rerun()

    # Chat history
    for turn in st.session_state.history:
        st.markdown(f"<div class='user-bubble'>🧑 {turn['user']}</div>", unsafe_allow_html=True)

        live_models  = turn["models"]
        show_nandi_live = turn.get("nandi_live", False)
        all_cols = st.columns(1 + len(live_models))

        # Nandi column
        with all_cols[0]:
            nandi_r = turn["results"].get("Nandi-600M ⭐", {})
            st.markdown(
                "<div class='model-header' style='background:#FF6B3522;"
                "border-color:#FF6B35;color:#FF6B35;'>"
                "Nandi-600M ⭐<br>"
                "<span style='font-weight:400;color:#8b949e'>600M · 20% trained · Indic-optimised</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            if show_nandi_live and nandi_r:
                if nandi_r.get("error"):
                    st.markdown(f"<div class='error-box'>{nandi_r['error'][:200]}</div>",
                                unsafe_allow_html=True)
                else:
                    st.markdown(
                        f"<div class='model-output'>{nandi_r.get('text','') or '(no output)'}</div>",
                        unsafe_allow_html=True,
                    )
                    tps = nandi_r["tokens"] / nandi_r["time"] if nandi_r["time"] > 0 else 0
                    st.markdown(
                        f"<div class='timing'>⏱ {nandi_r['time']:.2f}s · "
                        f"{nandi_r['tokens']} tok · {tps:.0f} tok/s</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    "<div class='nandi-local'>"
                    "🖥️ Local inference required<br>"
                    "<small>Add endpoint URL in sidebar<br>to run Nandi live</small>"
                    "</div>",
                    unsafe_allow_html=True,
                )

        # Comparison model outputs
        for i, mname in enumerate(live_models):
            r   = turn["results"].get(mname, {})
            cfg = MODELS.get(mname, {})
            color = cfg.get("color", "#888")
            with all_cols[i + 1]:
                st.markdown(
                    f"<div class='model-header' style='background:{color}22;"
                    f"border-color:{color};color:{color};'>"
                    f"{mname}<br>"
                    f"<span style='font-weight:400;color:#8b949e'>"
                    f"{cfg.get('params','')} · {cfg.get('note','')}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if r.get("error"):
                    st.markdown(f"<div class='error-box'>{r['error'][:200]}</div>",
                                unsafe_allow_html=True)
                else:
                    st.markdown(
                        f"<div class='model-output'>{r.get('text','') or '(no output)'}</div>",
                        unsafe_allow_html=True,
                    )
                    tps = r["tokens"] / r["time"] if r.get("time", 0) > 0 else 0
                    st.markdown(
                        f"<div class='timing'>⏱ {r.get('time',0):.2f}s · "
                        f"{r.get('tokens',0)} tok · {tps:.0f} tok/s</div>",
                        unsafe_allow_html=True,
                    )
        st.divider()

    # Input
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "msg", placeholder="Ask in English, Telugu, Hindi, or any language…",
            height=80, label_visibility="collapsed",
        )
        submitted = st.form_submit_button("🚀  Send to all models")

    if submitted:
        if not groq_key and not nandi_url:
            st.error("Add a Groq API key in the sidebar.")
        elif not selected_models and not nandi_url:
            st.error("Select at least one model.")
        elif not user_input.strip():
            st.warning("Type something first.")
        else:
            with st.spinner(f"Calling {len(selected_models) + (1 if nandi_url else 0)} models…"):
                res = run_all(selected_models, user_input.strip(), max_tokens,
                              temperature, groq_key, sarvam_key, nandi_url)
            st.session_state.history.append({
                "user": user_input.strip(),
                "results": res,
                "models": list(selected_models),
                "nandi_live": nandi_live,
            })
            st.rerun()


# ═══════════════════════════════ LIVE METRICS ═════════════════════════════════
with tab_metrics:
    st.subheader("Live Performance Metrics")

    if not st.session_state.history:
        st.info("Run a query in the Live Arena first — metrics will appear here.")
    else:
        # Aggregate all runs
        all_rows = []
        for turn in st.session_state.history:
            for mname, r in turn["results"].items():
                if not r.get("error") and r.get("time", 0) > 0:
                    tps = r["tokens"] / r["time"] if r["time"] > 0 else 0
                    all_rows.append({
                        "Model":         mname,
                        "Response Time (s)": round(r["time"], 2),
                        "Output Tokens": r.get("tokens", 0),
                        "Tokens / sec":  round(tps, 1),
                        "Word Count":    len(r.get("text", "").split()),
                        "Chars":         len(r.get("text", "")),
                    })

        if all_rows:
            df_live = pd.DataFrame(all_rows)
            avg = df_live.groupby("Model").mean(numeric_only=True).round(1).reset_index()

            # Colour map
            cmap = {n: cfg["color"] for n, cfg in MODELS.items()}
            cmap["Nandi-600M ⭐"] = "#FF6B35"

            st.markdown("##### Average across all queries so far")

            c1, c2, c3 = st.columns(3)

            # Response Time
            with c1:
                fig_t = go.Figure(go.Bar(
                    x=avg["Model"], y=avg["Response Time (s)"],
                    marker_color=[cmap.get(n, "#888") for n in avg["Model"]],
                    text=avg["Response Time (s)"].apply(lambda v: f"{v:.1f}s"),
                    textposition="outside",
                ))
                fig_t.update_layout(
                    title="⏱ Response Time (lower = better)",
                    yaxis_title="Seconds",
                    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                    font_color="#e6edf3", height=300, margin=dict(t=40, b=10),
                    xaxis_tickangle=-30,
                )
                st.plotly_chart(fig_t, use_container_width=True)

            # Tokens per second
            with c2:
                fig_s = go.Figure(go.Bar(
                    x=avg["Model"], y=avg["Tokens / sec"],
                    marker_color=[cmap.get(n, "#888") for n in avg["Model"]],
                    text=avg["Tokens / sec"].apply(lambda v: f"{v:.0f}"),
                    textposition="outside",
                ))
                fig_s.update_layout(
                    title="⚡ Tokens / Second (higher = better)",
                    yaxis_title="tok/s",
                    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                    font_color="#e6edf3", height=300, margin=dict(t=40, b=10),
                    xaxis_tickangle=-30,
                )
                st.plotly_chart(fig_s, use_container_width=True)

            # Output quality proxy: word count
            with c3:
                fig_w = go.Figure(go.Bar(
                    x=avg["Model"], y=avg["Word Count"],
                    marker_color=[cmap.get(n, "#888") for n in avg["Model"]],
                    text=avg["Word Count"].apply(lambda v: f"{v:.0f}"),
                    textposition="outside",
                ))
                fig_w.update_layout(
                    title="📝 Avg Output Words",
                    yaxis_title="Words",
                    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                    font_color="#e6edf3", height=300, margin=dict(t=40, b=10),
                    xaxis_tickangle=-30,
                )
                st.plotly_chart(fig_w, use_container_width=True)

            st.divider()
            st.markdown("##### Full metrics table (averages)")
            st.dataframe(avg.set_index("Model"), use_container_width=True)

            # Winner badges
            st.divider()
            st.markdown("##### 🏆 Leaders")
            b1, b2, b3 = st.columns(3)
            fastest = avg.loc[avg["Response Time (s)"].idxmin(), "Model"]
            fastest_tps = avg.loc[avg["Tokens / sec"].idxmax(), "Model"]
            most_words = avg.loc[avg["Word Count"].idxmax(), "Model"]
            b1.metric("⏱ Fastest response", fastest,
                      f"{avg.loc[avg['Model']==fastest,'Response Time (s)'].values[0]:.1f}s")
            b2.metric("⚡ Highest throughput", fastest_tps,
                      f"{avg.loc[avg['Model']==fastest_tps,'Tokens / sec'].values[0]:.0f} tok/s")
            b3.metric("📝 Most detailed output", most_words,
                      f"{avg.loc[avg['Model']==most_words,'Word Count'].values[0]:.0f} words avg")

    st.divider()
    # Static section always visible
    st.subheader("Tokenisation Fertility — Nandi vs Peers (static reference)")
    st.caption("Lower fertility = fewer tokens per word = more efficient Indic handling")

    fig_f = go.Figure()
    pal = {"Nandi-600M": "#FF6B35", "Qwen3-0.6B": "#3498DB", "SmolLM3-3B": "#9B59B6"}
    for col in pal:
        fig_f.add_trace(go.Bar(
            name=col, x=FERTILITY["Language"], y=FERTILITY[col],
            marker_color=pal[col], text=FERTILITY[col], textposition="outside",
        ))
    fig_f.update_layout(
        barmode="group", yaxis_title="Fertility (lower = better)",
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font_color="#e6edf3", height=320, legend=dict(bgcolor="#0d1117"),
        margin=dict(t=10),
    )
    st.plotly_chart(fig_f, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Benchmark scores** (* early checkpoint at 20% training)")
        st.dataframe(BENCHMARKS, use_container_width=True, hide_index=True)
    with col_b:
        ratios = (FERTILITY["Qwen3-0.6B"] / FERTILITY["Nandi-600M"]).tolist()
        fig_r = go.Figure(go.Bar(
            x=FERTILITY["Language"].tolist(), y=ratios,
            marker_color=["#FF6B35"] * 5,
            text=[f"{r:.1f}×" for r in ratios], textposition="outside",
        ))
        fig_r.update_layout(
            title="Telugu/Indic efficiency vs Qwen3 (higher = Nandi advantage)",
            yaxis_title="Efficiency ratio",
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font_color="#e6edf3", height=300, margin=dict(t=40),
        )
        st.plotly_chart(fig_r, use_container_width=True)


# ═══════════════════════════════ INDIC EDGE ═══════════════════════════════════
with tab_edge:
    st.subheader("Nandi's Indic Edge — Architecture & Benchmarks")
    st.markdown(
        "**Fertility** = tokens per word. At 1.77 for Telugu, Nandi needs 7.5× fewer tokens "
        "than Qwen3-0.6B (13.38). This is a structural, architectural advantage that "
        "compounds as training scales."
    )

    fig = go.Figure()
    for col in pal:
        fig.add_trace(go.Bar(
            name=col, x=FERTILITY["Language"], y=FERTILITY[col],
            marker_color=pal[col], text=FERTILITY[col], textposition="outside",
        ))
    fig.update_layout(
        barmode="group", yaxis_title="Fertility score (lower = better)",
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font_color="#e6edf3", height=400, legend=dict(bgcolor="#0d1117"),
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Benchmark scores  (* early checkpoint)**")
        st.dataframe(BENCHMARKS, use_container_width=True, hide_index=True)
        st.caption("At 20% training, Nandi is within ~7% of models trained on 16–180× more data.")
    with c2:
        fig2 = go.Figure(go.Bar(
            x=FERTILITY["Language"].tolist(), y=ratios,
            marker_color=["#FF6B35"] * 5,
            text=[f"{r:.1f}×" for r in ratios], textposition="outside",
        ))
        fig2.update_layout(
            title="Nandi efficiency advantage over Qwen3",
            yaxis_title="Tokens saved per word",
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font_color="#e6edf3", height=320, margin=dict(t=40),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.info(
        "📌 **Investor note:** Nandi's tokenisation efficiency is architectural — "
        "it persists and compounds at full training. 7.5× fewer tokens for Telugu = "
        "7.5× more context per window + proportionally lower per-query inference cost."
    )


# ═══════════════════════════════ RUN LOCALLY ══════════════════════════════════
with tab_local:
    st.subheader("Running Nandi locally and connecting it to this arena")

    st.markdown("""
Once Nandi is running locally, paste the endpoint URL into the sidebar — it will appear
as a **live column** in every query alongside the cloud models.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Option A — Docker model runner (one command)**")
        st.code(
            "docker model run hf.co/FrontiersMind/Nandi-Mini-600M-Early-Checkpoint",
            language="bash",
        )
        st.caption("Starts an OpenAI-compatible server. Then paste `http://localhost:12434` in the sidebar.")

        st.markdown("**Option B — vLLM**")
        st.code(
            "pip install vllm\n"
            "vllm serve FrontiersMind/Nandi-Mini-600M-Early-Checkpoint "
            "--trust-remote-code --dtype bfloat16",
            language="bash",
        )
        st.caption("Then paste `http://localhost:8000` in the sidebar.")

    with col2:
        st.markdown("**Option C — Python (Transformers, no server)**")
        st.code(LOCAL_CODE, language="python")

        st.markdown("**Model specs**")
        st.markdown("""
| Property | Value |
|---|---|
| Parameters | 600M |
| KV Cache | Shared (50% memory saving) |
| Context length | 2,048 → 32K planned |
| Languages | English + 10 Indic |
| Training | 250B / ~1.25T tokens (20%) |
| Precision | BF16 |
| License | Apache 2.0 |
        """)


# ═══════════════════════════════ ABOUT ════════════════════════════════════════
with tab_about:
    st.subheader("Live comparison models")
    for name, cfg in MODELS.items():
        with st.expander(f"{name} — {cfg['params']} · {cfg['note']}"):
            st.code(cfg["id"], language=None)
            if cfg["provider"] == "sarvam":
                st.write("Provider: **Sarvam AI** — [dashboard.sarvam.ai](https://dashboard.sarvam.ai)")
            else:
                st.write("Provider: **Groq** — [console.groq.com](https://console.groq.com) (free)")

    st.divider()
    st.markdown("""
**Getting API keys**

| Key | Where | Cost |
|---|---|---|
| Groq | [console.groq.com](https://console.groq.com) | Free, no CC |
| Sarvam | [dashboard.sarvam.ai](https://dashboard.sarvam.ai) | Freemium |

**Running Nandi live** — see the **Run Nandi Locally** tab. Once running, paste the base URL
(e.g. `http://localhost:8000`) into the sidebar. It will join the arena as a live column.
    """)
