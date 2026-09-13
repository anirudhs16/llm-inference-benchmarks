# LLM Inference Optimization Dashboard

Benchmark any LLM across backends and output configurations. Get P50/P95 latency, throughput, cost per 1K tokens, and a recommendation — in one command.

**Supports:** Ollama · vLLM · OpenAI API · Anthropic API

---

## Quick start

```bash
git clone https://github.com/anirudhs16/llm-inference-benchmarks.git
cd llm-inference-benchmarks
pip install -r requirements.txt

# Ollama (local — no API key needed)
ollama serve
python dashboard.py --model llama3.2
```

**Sample output:**

```
================================================================================
  LLM INFERENCE OPTIMIZATION DASHBOARD — llama3.2
================================================================================
  Config                 Backend      P50 TTFT  P95 TTFT   tok/s    $/1K  Recommendation
  ----------------------------------------------------------------------------
  short-output           ollama         2.41s     7.46s    14.2    free  ✅ Good — interactive chat
  medium-output          ollama         2.55s     7.80s    12.4    free  ✅ Good — interactive chat
  long-output            ollama         3.10s     8.20s     9.8    free  ⚡ Acceptable — streaming helps
================================================================================

  🏆 Best latency:    short-output [ollama] — P50 2.41s
  🚀 Best throughput: short-output [ollama] — 14.2 tok/s
```

---

## All backends

### Ollama (local, free)
```bash
ollama serve
ollama pull llama3.2
python dashboard.py --model llama3.2 --backend ollama
```

### vLLM (self-hosted GPU)
```bash
# Start vLLM server first
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --port 8000

# Run dashboard
python dashboard.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --backend vllm \
    --vllm-url http://localhost:8000
```

### OpenAI API
```bash
python dashboard.py \
    --model gpt-4o-mini \
    --backend openai \
    --api-key sk-...

# Or set env var
export OPENAI_API_KEY=sk-...
python dashboard.py --model gpt-4o-mini --backend openai
```

### Anthropic API
```bash
python dashboard.py \
    --model claude-haiku-4-5-20251001 \
    --backend anthropic \
    --api-key sk-ant-...

# Or set env var
export ANTHROPIC_API_KEY=sk-ant-...
python dashboard.py --model claude-haiku-4-5-20251001 --backend anthropic
```

---

## Compare backends

Run the dashboard twice with `--save` and compare the JSON outputs:

```bash
# Run Ollama
python dashboard.py --model llama3.2 --backend ollama --save --output ollama_results.json

# Run vLLM (same model, GPU)
python dashboard.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --backend vllm \
    --save --output vllm_results.json
```

---

## All flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `llama3.2` | Model name |
| `--backend` | `ollama` | `ollama`, `vllm`, `openai`, `anthropic` |
| `--runs` | `3` | Requests per config |
| `--prompt-type` | `short` | `short`, `medium`, `long` |
| `--ollama-url` | `http://localhost:11434` | Ollama server URL |
| `--vllm-url` | `http://localhost:8000` | vLLM server URL |
| `--api-key` | `None` | API key (or use env var) |
| `--cost` | auto | Override cost per 1K tokens |
| `--save` | off | Save JSON report |
| `--output` | auto-named | JSON output filename |

---

## Metrics explained

| Metric | What it means | Why it matters |
|--------|--------------|----------------|
| P50 TTFT | Median time to first token | Typical user experience |
| P95 TTFT | 95th percentile TTFT | Worst case — includes cold starts |
| tok/s | Tokens per second (decode phase) | Throughput for long responses |
| $/1K tokens | Cost per 1000 tokens | Budget planning |

**Why P50/P95, not average:**
Run 1 is always slower — the model loads into memory (cold start). P50 shows the warm experience most users see. P95 catches cold starts and outliers that show up in SLAs. Averaging them hides both.

---

## Cost defaults

| Backend | Default cost/1K tokens |
|---------|----------------------|
| Ollama | $0.00 (local) |
| vLLM | $0.00 (self-hosted) |
| OpenAI | $0.0015 (gpt-4o-mini estimate) |
| Anthropic | $0.00125 (claude-haiku estimate) |

Override with `--cost 0.002` for any backend.

---

## JSON report format

```json
{
  "timestamp": "2026-09-05 10:30:00",
  "results": [
    {
      "config": "short-output",
      "backend": "ollama",
      "model": "llama3.2",
      "p50_ttft_s": 2.41,
      "p95_ttft_s": 7.46,
      "avg_tokens_per_sec": 14.2,
      "avg_total_time_s": 8.62,
      "cost_per_1k_tokens": 0.0,
      "recommendation": "✅ Good — interactive chat"
    }
  ]
}
```

---

## How this was built

Built after 12 weeks of benchmarking LLM inference from the ground up:
- Measured Ollama vs vLLM concurrent latency (99× speedup)
- Profiled PyTorch kernel calls (1460 calls for 20 tokens)
- Benchmarked quantization across FP16/INT8/INT4
- Measured prefill vs decode phase separation
- Wrote Triton GPU kernels to understand what runs under the hood

Full benchmark series: [anirudhs16.github.io/ai-systems](https://anirudhs16.github.io/ai-systems)

---

## Requirements

```
requests>=2.31.0
```

No other dependencies. Pure Python stdlib + requests.
