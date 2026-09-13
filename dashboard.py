"""
LLM Inference Optimization Dashboard
Benchmarks LLMs across backends and configurations.

Supported backends:
  --backend ollama   Local Ollama (default)
  --backend vllm     vLLM OpenAI-compatible API
  --backend openai   OpenAI API
  --backend anthropic Anthropic API

Usage:
  python dashboard.py --model llama3.2 --backend ollama
  python dashboard.py --model meta-llama/Llama-3.2-1B-Instruct --backend vllm --vllm-url http://localhost:8000
  python dashboard.py --model gpt-4o-mini --backend openai --api-key sk-...
  python dashboard.py --model claude-haiku-4-5-20251001 --backend anthropic --api-key sk-ant-...
"""

import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass
from typing import Optional

import requests

# ── Prompts ───────────────────────────────────────────────────────────────────

PROMPTS = {
    "short":  "What is a KV cache? One sentence.",
    "medium": "Explain KV cache and PagedAttention and why they matter for inference.",
    "long":   "Write a detailed explanation of LLM inference optimization covering KV cache, quantization, continuous batching, speculative decoding, and PagedAttention.",
}

# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    ttft: float
    total_time: float
    tokens_generated: int
    tokens_per_sec: float


@dataclass
class BenchmarkResult:
    config: str
    backend: str
    model: str
    p50_ttft: float
    p95_ttft: float
    avg_tokens_per_sec: float
    avg_total_time: float
    cost_per_1k: float
    recommendation: str


# ── Backend: Ollama ───────────────────────────────────────────────────────────

def run_ollama(model: str, prompt: str, num_predict: int, url: str) -> Optional[RunResult]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"num_predict": num_predict},
    }
    first_token_time = None
    token_count = 0
    start = time.time()
    try:
        with requests.post(f"{url}/api/generate", json=payload, stream=True, timeout=120) as r:
            for line in r.iter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("response", "")
                    if first_token_time is None and token:
                        first_token_time = time.time() - start
                    token_count += 1
                    if data.get("done"):
                        break
    except Exception as e:
        print(f"    [ollama error] {e}")
        return None

    total_time = time.time() - start
    decode_time = total_time - (first_token_time or total_time)
    return RunResult(
        ttft=first_token_time or total_time,
        total_time=total_time,
        tokens_generated=token_count,
        tokens_per_sec=token_count / decode_time if decode_time > 0 else 0,
    )


# ── Backend: vLLM (OpenAI-compatible) ────────────────────────────────────────

def run_vllm(model: str, prompt: str, num_predict: int, url: str) -> Optional[RunResult]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": num_predict,
        "stream": True,
    }
    headers = {"Content-Type": "application/json"}
    first_token_time = None
    token_count = 0
    start = time.time()
    try:
        with requests.post(
            f"{url}/v1/chat/completions",
            json=payload,
            headers=headers,
            stream=True,
            timeout=120,
        ) as r:
            for line in r.iter_lines():
                if line:
                    raw = line.decode("utf-8")
                    if raw.startswith("data: "):
                        raw = raw[6:]
                    if raw.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(raw)
                        delta = data["choices"][0]["delta"].get("content", "")
                        if delta:
                            if first_token_time is None:
                                first_token_time = time.time() - start
                            token_count += 1
                    except Exception:
                        pass
    except Exception as e:
        print(f"    [vllm error] {e}")
        return None

    total_time = time.time() - start
    decode_time = total_time - (first_token_time or total_time)
    return RunResult(
        ttft=first_token_time or total_time,
        total_time=total_time,
        tokens_generated=token_count,
        tokens_per_sec=token_count / decode_time if decode_time > 0 else 0,
    )


# ── Backend: OpenAI ───────────────────────────────────────────────────────────

def run_openai(model: str, prompt: str, num_predict: int, api_key: str) -> Optional[RunResult]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": num_predict,
        "stream": True,
    }
    first_token_time = None
    token_count = 0
    start = time.time()
    try:
        with requests.post(
            "https://api.openai.com/v1/chat/completions",
            json=payload,
            headers=headers,
            stream=True,
            timeout=120,
        ) as r:
            for line in r.iter_lines():
                if line:
                    raw = line.decode("utf-8")
                    if raw.startswith("data: "):
                        raw = raw[6:]
                    if raw.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(raw)
                        delta = data["choices"][0]["delta"].get("content", "")
                        if delta:
                            if first_token_time is None:
                                first_token_time = time.time() - start
                            token_count += 1
                    except Exception:
                        pass
    except Exception as e:
        print(f"    [openai error] {e}")
        return None

    total_time = time.time() - start
    decode_time = total_time - (first_token_time or total_time)
    return RunResult(
        ttft=first_token_time or total_time,
        total_time=total_time,
        tokens_generated=token_count,
        tokens_per_sec=token_count / decode_time if decode_time > 0 else 0,
    )


# ── Backend: Anthropic ────────────────────────────────────────────────────────

def run_anthropic(model: str, prompt: str, num_predict: int, api_key: str) -> Optional[RunResult]:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": num_predict,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    first_token_time = None
    token_count = 0
    start = time.time()
    try:
        with requests.post(
            "https://api.anthropic.com/v1/messages",
            json=payload,
            headers=headers,
            stream=True,
            timeout=120,
        ) as r:
            for line in r.iter_lines():
                if line:
                    raw = line.decode("utf-8")
                    if raw.startswith("data: "):
                        raw = raw[6:]
                    try:
                        data = json.loads(raw)
                        if data.get("type") == "content_block_delta":
                            delta = data.get("delta", {}).get("text", "")
                            if delta:
                                if first_token_time is None:
                                    first_token_time = time.time() - start
                                token_count += 1
                    except Exception:
                        pass
    except Exception as e:
        print(f"    [anthropic error] {e}")
        return None

    total_time = time.time() - start
    decode_time = total_time - (first_token_time or total_time)
    return RunResult(
        ttft=first_token_time or total_time,
        total_time=total_time,
        tokens_generated=token_count,
        tokens_per_sec=token_count / decode_time if decode_time > 0 else 0,
    )


# ── Run one backend request (dispatcher) ──────────────────────────────────────

def run_single(backend: str, model: str, prompt: str, num_predict: int, args) -> Optional[RunResult]:
    if backend == "ollama":
        return run_ollama(model, prompt, num_predict, args.ollama_url)
    elif backend == "vllm":
        return run_vllm(model, prompt, num_predict, args.vllm_url)
    elif backend == "openai":
        key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
        return run_openai(model, prompt, num_predict, key)
    elif backend == "anthropic":
        key = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        return run_anthropic(model, prompt, num_predict, key)
    return None


# ── Cost table (per 1K tokens) ────────────────────────────────────────────────

COST_TABLE = {
    "ollama":    0.0,
    "vllm":      0.0,      # self-hosted — you pay for GPU, not per token
    "openai":    0.0015,   # rough gpt-4o-mini blended; override with --cost
    "anthropic": 0.00125,  # claude-haiku blended; override with --cost
}


# ── Benchmark one config ──────────────────────────────────────────────────────

def benchmark_config(
    backend: str, model: str, config_name: str,
    prompt_type: str, num_predict: int, runs: int, args
) -> Optional[BenchmarkResult]:

    prompt = PROMPTS[prompt_type]
    print(f"\n  [{backend}] {config_name} | prompt={prompt_type} | max_tokens={num_predict}")

    results = []
    for i in range(runs):
        r = run_single(backend, model, prompt, num_predict, args)
        if r:
            results.append(r)
            print(f"    run {i+1}/{runs}: TTFT={r.ttft:.2f}s | {r.tokens_per_sec:.1f} tok/s | {r.tokens_generated} tokens")

    if not results:
        return None

    ttfts = sorted(r.ttft for r in results)

    def pct(data, p):
        idx = int(len(data) * p / 100)
        return data[min(idx, len(data) - 1)]

    p50 = pct(ttfts, 50)
    p95 = pct(ttfts, 95)
    avg_tps = statistics.mean(r.tokens_per_sec for r in results)
    avg_total = statistics.mean(r.total_time for r in results)
    avg_tokens = statistics.mean(r.tokens_generated for r in results)
    cost_per_1k = args.cost if args.cost is not None else COST_TABLE.get(backend, 0.0)
    cost = (avg_tokens / 1000) * cost_per_1k

    if p50 < 2.0 and avg_tps > 15:
        rec = "✅ Excellent — real-time interactive"
    elif p50 < 4.0 and avg_tps > 8:
        rec = "✅ Good — interactive chat"
    elif p50 < 8.0:
        rec = "⚡ Acceptable — streaming helps"
    else:
        rec = "⚠️  Batch processing recommended"

    return BenchmarkResult(
        config=config_name,
        backend=backend,
        model=model,
        p50_ttft=p50,
        p95_ttft=p95,
        avg_tokens_per_sec=avg_tps,
        avg_total_time=avg_total,
        cost_per_1k=cost_per_1k,
        recommendation=rec,
    )


# ── Print results ─────────────────────────────────────────────────────────────

def print_results(results: list[BenchmarkResult]):
    if not results:
        print("No results to display.")
        return

    model = results[0].model
    print("\n")
    print("=" * 80)
    print(f"  LLM INFERENCE OPTIMIZATION DASHBOARD — {model}")
    print("=" * 80)
    print(f"  {'Config':<22} {'Backend':<12} {'P50 TTFT':>9} {'P95 TTFT':>9} {'tok/s':>7} {'$/1K':>7}  Recommendation")
    print("  " + "-" * 76)
    for r in results:
        cost_str = f"${r.cost_per_1k:.4f}" if r.cost_per_1k > 0 else "free"
        print(
            f"  {r.config:<22} {r.backend:<12} "
            f"{r.p50_ttft:>8.2f}s {r.p95_ttft:>8.2f}s "
            f"{r.avg_tokens_per_sec:>6.1f} {cost_str:>7}  {r.recommendation}"
        )
    print("=" * 80)

    best_latency = min(results, key=lambda r: r.p50_ttft)
    best_throughput = max(results, key=lambda r: r.avg_tokens_per_sec)
    print(f"\n  🏆 Best latency:    {best_latency.config} [{best_latency.backend}] — P50 {best_latency.p50_ttft:.2f}s")
    print(f"  🚀 Best throughput: {best_throughput.config} [{best_throughput.backend}] — {best_throughput.avg_tokens_per_sec:.1f} tok/s")
    if any(r.cost_per_1k > 0 for r in results):
        cheapest = min((r for r in results if r.cost_per_1k > 0), key=lambda r: r.cost_per_1k)
        print(f"  💰 Cheapest:        {cheapest.config} [{cheapest.backend}] — ${cheapest.cost_per_1k:.4f}/1K tokens")
    print()


# ── Save report ───────────────────────────────────────────────────────────────

def save_report(results: list[BenchmarkResult], filename: str):
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [
            {
                "config": r.config,
                "backend": r.backend,
                "model": r.model,
                "p50_ttft_s": round(r.p50_ttft, 3),
                "p95_ttft_s": round(r.p95_ttft, 3),
                "avg_tokens_per_sec": round(r.avg_tokens_per_sec, 1),
                "avg_total_time_s": round(r.avg_total_time, 3),
                "cost_per_1k_tokens": r.cost_per_1k,
                "recommendation": r.recommendation,
            }
            for r in results
        ],
    }
    with open(filename, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved → {filename}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LLM Inference Optimization Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ollama (local)
  python dashboard.py --model llama3.2

  # vLLM (start with: python -m vllm.entrypoints.openai.api_server --model <model>)
  python dashboard.py --model meta-llama/Llama-3.2-1B-Instruct --backend vllm

  # OpenAI
  python dashboard.py --model gpt-4o-mini --backend openai --api-key sk-...

  # Anthropic
  python dashboard.py --model claude-haiku-4-5-20251001 --backend anthropic --api-key sk-ant-...

  # Compare two backends (run twice, results saved to JSON)
  python dashboard.py --model llama3.2 --backend ollama --save --output ollama.json
  python dashboard.py --model llama3.2 --backend vllm   --save --output vllm.json
        """,
    )
    parser.add_argument("--model",       default="llama3.2")
    parser.add_argument("--backend",     default="ollama", choices=["ollama", "vllm", "openai", "anthropic"])
    parser.add_argument("--runs",        type=int, default=3)
    parser.add_argument("--prompt-type", default="short", choices=["short", "medium", "long"])
    parser.add_argument("--ollama-url",  default="http://localhost:11434")
    parser.add_argument("--vllm-url",    default="http://localhost:8000")
    parser.add_argument("--api-key",     default=None, help="API key (or set OPENAI_API_KEY / ANTHROPIC_API_KEY env var)")
    parser.add_argument("--cost",        type=float, default=None, help="Override cost per 1K tokens")
    parser.add_argument("--save",        action="store_true")
    parser.add_argument("--output",      default=None, help="Output JSON filename")
    args = parser.parse_args()

    print(f"\n🔍 LLM Inference Dashboard")
    print(f"   model={args.model} | backend={args.backend} | runs={args.runs} | prompt={args.prompt_type}")

    # Configs: vary output length to show prefill vs decode tradeoff
    configs = [
        ("short-output",  50),
        ("medium-output", 200),
        ("long-output",   500),
    ]

    results = []
    for config_name, num_predict in configs:
        r = benchmark_config(
            args.backend, args.model, config_name,
            args.prompt_type, num_predict, args.runs, args
        )
        if r:
            results.append(r)

    print_results(results)

    if args.save:
        filename = args.output or f"report_{args.backend}_{args.model.replace('/', '-').replace(':', '-')}_{int(time.time())}.json"
        save_report(results, filename)


if __name__ == "__main__":
    main()
