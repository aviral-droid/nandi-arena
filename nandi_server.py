"""
Nandi local inference server — OpenAI-compatible API
Runs on http://localhost:8080
Paste that URL into the Nandi Arena sidebar to run Nandi live.
"""

import time
import uuid
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_ID = "FrontiersMind/Nandi-Mini-600M-Early-Checkpoint"
PORT = 8080

print(f"\n⚡ Loading Nandi ({MODEL_ID})...")
print("   First run downloads ~1.2 GB — subsequent runs are instant.\n")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Use MPS on Apple Silicon, CPU otherwise
if torch.backends.mps.is_available():
    device = "mps"
    dtype  = torch.float16
    print("   Using Apple Silicon GPU (MPS) ✓")
else:
    device = "cpu"
    dtype  = torch.float32
    print("   Using CPU")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, trust_remote_code=True, torch_dtype=dtype
).to(device).eval()
model.config.kv_cache_mode = "shared"   # 50% memory saving

print(f"\n✅ Nandi ready — server starting at http://localhost:{PORT}\n")
print("   Paste  http://localhost:8080  into the Nandi Arena sidebar.\n")

app = Flask(__name__)


def _generate(prompt, max_tokens, temperature):
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    n_prompt = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=max(float(temperature), 0.01),
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][n_prompt:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text, n_prompt, len(new_tokens)


@app.route("/v1/completions", methods=["POST"])
def completions():
    data       = request.get_json(force=True)
    prompt     = data.get("prompt", "")
    max_tokens = int(data.get("max_tokens", 200))
    temp       = data.get("temperature", 0.3)
    text, n_in, n_out = _generate(prompt, max_tokens, temp)
    return jsonify({
        "id":      f"cmpl-{uuid.uuid4().hex[:8]}",
        "object":  "text_completion",
        "created": int(time.time()),
        "model":   MODEL_ID,
        "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
        "usage":   {"prompt_tokens": n_in, "completion_tokens": n_out,
                    "total_tokens": n_in + n_out},
    })


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data       = request.get_json(force=True)
    messages   = data.get("messages", [])
    prompt     = messages[-1]["content"] if messages else ""
    max_tokens = int(data.get("max_tokens", 200))
    temp       = data.get("temperature", 0.3)
    text, n_in, n_out = _generate(prompt, max_tokens, temp)
    return jsonify({
        "id":      f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   MODEL_ID,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": text}}],
        "usage":   {"prompt_tokens": n_in, "completion_tokens": n_out,
                    "total_tokens": n_in + n_out},
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_ID})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
