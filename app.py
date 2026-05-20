"""
Nandi LLM Arena — Investment Committee Benchmark Tool
Concurrently tests Nandi-Mini-600M against peer small LLMs.
"""

import time
import concurrent.futures

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from huggingface_hub import InferenceClient

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nandi LLM Arena",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .model-badge {
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 13px;
    font-weight: 700;
    border-left: 5px solid;
  }
  .output-box {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px;
    min-height: 140px;
    font-size: 13px;
    line-height: 1.75;
    white-space: pre-wrap;
    font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
    color: #e6edf3;
  }
  .timing {
    font-size: 11px;
    color: #8b949e;
    margin-top: 4px;
  }
  .highlight-border {
    border: 2px solid #FF6B35 !important;
    box-shadow: 0 0 12px #FF6B3540;
  }
  [data-testid="stButton"] button {
    background: linear-gradient(135deg, #FF6B35 0%, #f7931e 100%);
    color: white;
    font-weight: 700;
    font-size: 15px;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    width: 100%;
  }
</style>
""", unsafe_allow_html=True)

# ── Model registry ─────────────────────────────────────────────────────────────
# type: "base"     → uses text_generation endpoint, receives a completion-style prompt
# type: "instruct" → uses chat_completion endpoint, receives the instruction prompt
MODELS: dict[str, dict] = {
    "Nandi-600M ⭐": {
        "id": "FrontiersMind/Nandi-Mini-600M-Early-Checkpoint",
        "color": "#FF6B35",
        "type": "base",
        "params": "600M",
        "note": "250B tokens • 20% trained",
        "highlight": True,
    },
    "SmolLM2-360M": {
        "id": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "color": "#2ECC71",
        "type": "instruct",
        "params": "360M",
        "note": "4T tokens • HuggingFace",
        "highlight": False,
    },
    "Qwen2.5-0.5B": {
        "id": "Qwen/Qwen2.5-0.5B-Instruct",
        "color": "#3498DB",
        "type": "instruct",
        "params": "500M",
        "note": "Benchmark comparison",
        "highlight": False,
    },
    "SmolLM2-1.7B": {
        "id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "color": "#9B59B6",
        "type": "instruct",
        "params": "1.7B",
        "note": "3× size reference",
        "highlight": False,
    },
    "Qwen2.5-1.5B": {
        "id": "Qwen/Qwen2.5-1.5B-Instruct",
        "color": "#E74C3C",
        "type": "instruct",
        "params": "1.5B",
        "note": "Larger Qwen reference",
        "highlight": False,
    },
}

# ── Task presets ───────────────────────────────────────────────────────────────
# Each task has two prompt variants:
#   base_prompt    → completion-style (for Nandi and other base models)
#   instruct_prompt → instruction-style (for chat/instruct models)
TASKS: dict[str, dict] = {
    "🇮🇳 English → Telugu Translation": {
        "base_prompt": (
            "English: The technology sector in India is growing at an unprecedented rate, "
            "attracting billions in global investments every year.\n"
            "Telugu translation:"
        ),
        "instruct_prompt": (
            "Translate the following sentence into Telugu:\n\n"
            "The technology sector in India is growing at an unprecedented rate, "
            "attracting billions in global investments every year."
        ),
        "insight": (
            "Nandi's Telugu fertility score is **1.77** vs Qwen3's **13.38** — "
            "meaning Nandi needs 7.5× fewer tokens to encode the same Telugu text. "
            "This directly impacts generation quality and inference cost."
        ),
    },
    "🇮🇳 Telugu Story Continuation": {
        "base_prompt": (
            "తెలుగు కథ: అనగనగా కృష్ణా నది ఒడ్డున ఒక చిన్న పల్లె ఉండేది. "
            "ఆ పల్లెలో లక్ష్మి అనే తెలివైన అమ్మాయి నివసించేది. "
            "ఒక రోజు ఆమె"
        ),
        "instruct_prompt": (
            "Continue this Telugu story (write the continuation in Telugu only, 3-4 sentences):\n\n"
            "తెలుగు కథ: అనగనగా కృష్ణా నది ఒడ్డున ఒక చిన్న పల్లె ఉండేది. "
            "ఆ పల్లెలో లక్ష్మి అనే తెలివైన అమ్మాయి నివసించేది."
        ),
        "insight": (
            "Native Telugu generation fluency test. Nandi is trained on 11 Indic languages "
            "with an optimised tokeniser — expect significantly more natural Telugu output "
            "compared to English-first models."
        ),
    },
    "🌐 Bilingual Answer (English + Telugu)": {
        "base_prompt": (
            "Question: What are the main benefits of renewable energy for India?\n"
            "English Answer: Renewable energy offers India several critical advantages:\n"
            "Telugu Answer: పునరుత్పాదక శక్తి భారతదేశానికి కింది లాభాలు అందిస్తుంది:"
        ),
        "instruct_prompt": (
            "Answer the following question in BOTH English and Telugu.\n\n"
            "Question: What are the main benefits of renewable energy for India?\n\n"
            "Format your response as:\n"
            "English: [2-3 sentence answer]\n"
            "Telugu: [Telugu translation of the answer]"
        ),
        "insight": (
            "Dual-language output is directly relevant for India-focused investment reports "
            "that need to reach regional stakeholders. Tests both comprehension and bilingual fluency."
        ),
    },
    "💰 Financial Report Summary": {
        "base_prompt": (
            "Quarterly Financial Report — Q3 FY2025\n"
            "Revenue: ₹2,400 crore (+23% YoY). Operating margin: 18.5% (+180bps). "
            "Net profit: ₹312 crore. Digital services drove 60% of growth.\n"
            "Investor summary:"
        ),
        "instruct_prompt": (
            "Summarise this quarterly financial report in 3 concise bullet points for an investment committee:\n\n"
            "Revenue: ₹2,400 crore (+23% YoY). Operating margin: 18.5% (+180bps). "
            "Net profit: ₹312 crore. Digital services drove 60% of growth."
        ),
        "insight": (
            "Practical investment-committee use case: rapid summarisation of financial data. "
            "Tests domain-specific language comprehension and structured output."
        ),
    },
    "🧠 Business Reasoning": {
        "base_prompt": (
            "A startup has a monthly burn rate of ₹50 lakh, has raised ₹6 crore, "
            "and projects breaking even in 18 months. "
            "Q: How many months of runway does it have, and will it reach breakeven in time?\n"
            "A: The startup has"
        ),
        "instruct_prompt": (
            "A startup has:\n"
            "- Monthly burn rate: ₹50 lakh\n"
            "- Total funds raised: ₹6 crore\n"
            "- Projected breakeven: 18 months from now\n\n"
            "Calculate how many months of runway it has and whether it will reach breakeven "
            "before running out of funds. Show your working."
        ),
        "insight": (
            "Multi-step arithmetic + business reasoning. Tests whether small models can "
            "reliably handle financial word problems relevant to due diligence workflows."
        ),
    },
    "🇮🇳 English → Hindi Translation": {
        "base_prompt": (
            "English: India's startup ecosystem raised over $10 billion in venture capital "
            "funding during the first half of the year.\n"
            "Hindi translation:"
        ),
        "instruct_prompt": (
            "Translate the following into Hindi:\n\n"
            "India's startup ecosystem raised over $10 billion in venture capital "
            "funding during the first half of the year."
        ),
        "insight": (
            "Hindi is another core Indic language in Nandi's training corpus. "
            "Nandi's Hindi fertility is far lower than English-first models, "
            "enabling more precise and efficient Hindi generation."
        ),
    },
    "✏️ Custom Prompt": {
        "base_prompt": "",
        "instruct_prompt": "",
        "insight": "Enter any prompt to test all models simultaneously.",
    },
}

# ── Static data for the insights tab ─────────────────────────────────────────
FERTILITY = pd.DataFrame({
    "Language": ["Bengali", "Tamil", "Telugu", "Malayalam", "English"],
    "Nandi-600M": [1.44, 2.05, 1.77, 2.05, 1.18],
    "Qwen3-0.6B": [7.51, 10.93, 13.38, 14.56, 1.16],
    "SmolLM3-3B": [8.66, 13.56, 15.40, 17.77, 1.17],
})

BENCHMARKS = pd.DataFrame({
    "Model": ["Nandi-600M ⭐ *early*", "SmolLM2-360M", "Qwen3-0.6B"],
    "Tokens Trained": ["250B  (20%)", "4T", "36T"],
    "HellaSwag": [44.86, 56.30, 53.77],
    "WinoGrande": [54.77, 59.19, 59.19],
    "MMLU": [29.01, 25.55, 50.34],
    "Average": [44.10, 47.53, 49.75],
})

# ── Inference helpers ──────────────────────────────────────────────────────────
def call_model(
    name: str,
    cfg: dict,
    base_prompt: str,
    instruct_prompt: str,
    max_tokens: int,
    temperature: float,
    hf_token: str,
) -> dict:
    start = time.time()
    try:
        client = InferenceClient(model=cfg["id"], token=hf_token)
        prompt = base_prompt if cfg["type"] == "base" else instruct_prompt

        if not prompt.strip():
            return {"model": name, "text": "", "time": 0.0, "error": "Empty prompt"}

        if cfg["type"] == "instruct":
            resp = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=max(temperature, 0.01),
            )
            text = resp.choices[0].message.content or ""
        else:
            text = client.text_generation(
                prompt,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                repetition_penalty=1.1,
                do_sample=True,
            )

        return {
            "model": name,
            "text": text.strip(),
            "time": time.time() - start,
            "error": None,
        }
    except Exception as exc:
        return {
            "model": name,
            "text": "",
            "time": time.time() - start,
            "error": str(exc),
        }


def run_concurrent(
    selected: list[str],
    base_prompt: str,
    instruct_prompt: str,
    max_tokens: int,
    temperature: float,
    hf_token: str,
) -> dict:
    results: dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected)) as ex:
        futures = {
            ex.submit(
                call_model, name, MODELS[name],
                base_prompt, instruct_prompt,
                max_tokens, temperature, hf_token,
            ): name
            for name in selected
        }
        for fut in concurrent.futures.as_completed(futures):
            r = fut.result()
            results[r["model"]] = r
    return results


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Nandi Arena")
    st.caption("Investment Committee — LLM Benchmark")
    st.divider()

    # Read token from Streamlit secrets (for deployed app) or from user input
    _secret_token = st.secrets.get("HF_TOKEN", "") if hasattr(st, "secrets") else ""
    hf_token = st.text_input(
        "HuggingFace Token",
        value=_secret_token,
        type="password",
        placeholder="hf_...",
        help="Free token at huggingface.co/settings/tokens — read access is sufficient.",
    )
    if not hf_token:
        st.warning("Paste your HF token above to enable inference.")

    st.divider()
    st.markdown("**Models to compare**")
    selected_models: list[str] = []
    for name, cfg in MODELS.items():
        label = f"{name}  `{cfg['params']}`"
        if st.checkbox(label, value=True, key=f"chk_{name}"):
            selected_models.append(name)

    st.divider()
    st.markdown("**Generation settings**")
    max_tokens = st.slider("Max new tokens", 50, 500, 180, 25)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)

    st.divider()
    st.markdown(
        "<small>Nandi-Mini-600M is an **early pretraining checkpoint** at 20% training. "
        "Its tokenisation advantage is architecture-level and persists regardless of "
        "checkpoint maturity.</small>",
        unsafe_allow_html=True,
    )


# ── Main layout ────────────────────────────────────────────────────────────────
st.markdown("# ⚡ Nandi LLM Arena")
st.caption(
    "Side-by-side concurrent benchmark · Nandi-Mini-600M vs SmolLM2, Qwen2.5 & peers"
)

tab_compare, tab_insights, tab_about = st.tabs(
    ["🏁 Live Comparison", "📊 Nandi's Indic Edge", "ℹ️ About the Models"]
)

# ─────────────────────────────── TAB 1: COMPARISON ───────────────────────────
with tab_compare:
    task_name = st.selectbox("Task preset", list(TASKS.keys()))
    task = TASKS[task_name]

    st.info(f"💡 {task['insight']}")

    if task_name == "✏️ Custom Prompt":
        col1, col2 = st.columns(2)
        with col1:
            base_prompt = st.text_area(
                "Prompt for base models (Nandi)",
                height=130,
                placeholder="Write a completion-style prompt…",
                key="custom_base",
            )
        with col2:
            instruct_prompt = st.text_area(
                "Prompt for instruct models",
                height=130,
                placeholder="Write an instruction/question…",
                key="custom_instruct",
            )
    else:
        with st.expander("View / edit prompts", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                base_prompt = st.text_area(
                    "Base-model prompt",
                    value=task["base_prompt"],
                    height=130,
                    key="base_prompt",
                )
            with col2:
                instruct_prompt = st.text_area(
                    "Instruct-model prompt",
                    value=task["instruct_prompt"],
                    height=130,
                    key="instruct_prompt",
                )

    can_run = bool(hf_token and selected_models)
    run_btn = st.button(
        f"🚀  Run {len(selected_models)} Models Simultaneously",
        disabled=not can_run,
    )

    if run_btn and can_run:
        with st.spinner(f"Calling {len(selected_models)} models concurrently…"):
            results = run_concurrent(
                selected_models,
                base_prompt,
                instruct_prompt,
                max_tokens,
                temperature,
                hf_token,
            )

        st.divider()
        cols = st.columns(len(selected_models))

        for i, name in enumerate(selected_models):
            r = results.get(name, {})
            cfg = MODELS[name]
            with cols[i]:
                extra_class = "highlight-border" if cfg["highlight"] else ""
                st.markdown(
                    f"<div class='model-badge {extra_class}' "
                    f"style='background:{cfg['color']}18; border-color:{cfg['color']};'>"
                    f"<span style='color:{cfg['color']}'>{name}</span><br>"
                    f"<small style='color:#8b949e'>{cfg['params']} · {cfg['type']} · {cfg['note']}</small>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                if r.get("error"):
                    st.error(r["error"][:200])
                else:
                    output_text = r.get("text", "(no output)")
                    st.markdown(
                        f"<div class='output-box'>{output_text}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='timing'>⏱ {r['time']:.2f}s</div>",
                        unsafe_allow_html=True,
                    )

        # Timing bar chart
        st.divider()
        st.markdown("**Response times**")
        timing_data = {
            name: results[name]["time"]
            for name in selected_models
            if name in results and not results[name].get("error")
        }
        if timing_data:
            fig_t = go.Figure(go.Bar(
                x=list(timing_data.keys()),
                y=list(timing_data.values()),
                marker_color=[MODELS[n]["color"] for n in timing_data],
                text=[f"{v:.2f}s" for v in timing_data.values()],
                textposition="outside",
            ))
            fig_t.update_layout(
                yaxis_title="Seconds",
                plot_bgcolor="#0d1117",
                paper_bgcolor="#0d1117",
                font_color="#e6edf3",
                height=260,
                margin=dict(t=20, b=20),
                showlegend=False,
            )
            st.plotly_chart(fig_t, use_container_width=True)

    elif not hf_token:
        st.markdown(
            "<div style='text-align:center; padding:40px; color:#8b949e;'>"
            "Enter your HuggingFace token in the sidebar to run models.</div>",
            unsafe_allow_html=True,
        )


# ────────────────────────────── TAB 2: INSIGHTS ──────────────────────────────
with tab_insights:
    st.subheader("Tokenization Fertility — Nandi's Core Advantage")
    st.markdown(
        "**Fertility** = average tokens per word. Lower is better: the model encodes the "
        "same text in fewer tokens, reducing memory, inference cost, and information loss. "
        "Nandi's tokeniser is purpose-built for Indic scripts."
    )

    fig_f = go.Figure()
    palette = {"Nandi-600M": "#FF6B35", "Qwen3-0.6B": "#3498DB", "SmolLM3-3B": "#9B59B6"}
    for col in ["Nandi-600M", "Qwen3-0.6B", "SmolLM3-3B"]:
        fig_f.add_trace(go.Bar(
            name=col,
            x=FERTILITY["Language"],
            y=FERTILITY[col],
            marker_color=palette[col],
            text=FERTILITY[col],
            textposition="outside",
        ))
    fig_f.update_layout(
        barmode="group",
        yaxis_title="Fertility score (lower = better)",
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
        font_color="#e6edf3",
        height=420,
        legend=dict(bgcolor="#0d1117"),
    )
    st.plotly_chart(fig_f, use_container_width=True)

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Benchmark scores (* early checkpoint)")
        st.dataframe(BENCHMARKS, use_container_width=True, hide_index=True)
        st.caption(
            "Nandi is at 20% training. SmolLM2-360M used 4T tokens; "
            "Qwen3-0.6B used 36T. Trajectory suggests strong convergence at full training."
        )

    with col_b:
        st.subheader("Efficiency ratio: Telugu tokens")
        langs = FERTILITY["Language"].tolist()
        ratios = (FERTILITY["Qwen3-0.6B"] / FERTILITY["Nandi-600M"]).tolist()
        fig_r = go.Figure(go.Bar(
            x=langs,
            y=ratios,
            marker_color=["#FF6B35" if r > 3 else "#2ECC71" for r in ratios],
            text=[f"{r:.1f}×" for r in ratios],
            textposition="outside",
        ))
        fig_r.update_layout(
            title="Qwen3-0.6B tokens ÷ Nandi tokens (higher = Nandi is more efficient)",
            yaxis_title="Efficiency ratio",
            plot_bgcolor="#0d1117",
            paper_bgcolor="#0d1117",
            font_color="#e6edf3",
            height=340,
            margin=dict(t=40),
        )
        st.plotly_chart(fig_r, use_container_width=True)

    st.info(
        "📌 **Investment thesis context:** Nandi's tokenisation efficiency is a structural, "
        "architecture-level advantage that compounds as training scales. "
        "A 7.5× token reduction in Telugu means 7.5× more context fits in the same window, "
        "lower serving costs per Indian-language query, and better downstream accuracy on "
        "Indic-language tasks — all without any additional training."
    )


# ─────────────────────────────── TAB 3: ABOUT ────────────────────────────────
with tab_about:
    st.subheader("Models in this arena")

    for name, cfg in MODELS.items():
        with st.expander(f"{name} — {cfg['params']}"):
            st.markdown(f"**HuggingFace ID:** `{cfg['id']}`")
            st.markdown(f"**Type:** {cfg['type'].capitalize()} model")
            st.markdown(f"**Notes:** {cfg['note']}")
            if cfg["highlight"]:
                st.markdown(
                    "⭐ **This is the investment target.** Early pretraining checkpoint "
                    "(250B / ~1.25T planned tokens). Architecture: Transformer decoder "
                    "with Shared KV (50% memory saving), GQA, SwiGLU, RoPE. "
                    "Supports 11 languages: English + 10 Indic scripts."
                )

    st.divider()
    st.subheader("Prompt strategy")
    st.markdown("""
| Model type | Endpoint used | Prompt format |
|---|---|---|
| `base` (Nandi) | `text_generation` | Completion-style — model continues the text |
| `instruct` (others) | `chat_completion` | Instruction/question — model follows the request |

Both prompt variants are shown and editable in the **Live Comparison** tab.
Keeping two variants ensures a fair comparison: each model gets the input format it was trained for.
    """)

    st.subheader("Architecture highlights — Nandi")
    st.markdown("""
- **Shared KV cache** — reuses latent K/V projections, ~50% memory reduction vs vanilla MHA
- **Factorised tied embeddings** — compact vocabulary representation (131K tokens)
- **GQA + QK Norm + RMSNorm** — modern efficiency stack
- **RoPE positional encoding** — standard for long-context extension
- **Planned context:** 2,048 tokens now → 32,000 tokens at full training
    """)
